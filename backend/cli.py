"""WhisperMerger from a terminal (#36). One command per action in actions.py, built from each
function's signature and docstring, so the CLI and the MCP server always offer the same things.

    whispermerger transcribe a.mov --names "Host,Guest" --language en --out ./subs
    whispermerger get-transcript <job id>
    whispermerger models            (alias of list-models)

Lists are comma-separated for options (--names "Host,Guest"); file paths are plain arguments.
Progress goes to stderr; --json prints machine-readable results on stdout.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
import typing

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import actions  # noqa: E402

ALIASES = {"list-models": ["models"], "get-transcript": ["transcript"], "get-progress": ["progress"]}


def _param_help(fn) -> dict[str, str]:
    """`name: text` lines in the docstring become each option's help."""
    return {m[1]: m[2] for m in re.finditer(r"^\s*(\w+): (.+)$", inspect.getdoc(fn) or "", re.M)}


def _base(tp):
    """Optional[X] -> X."""
    args = [a for a in typing.get_args(tp) if a is not type(None)]
    return args[0] if typing.get_origin(tp) is typing.Union or type(tp).__name__ == "UnionType" else tp


def _comma(item_type):
    return lambda s: [item_type(x.strip()) for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="whispermerger", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="print results as JSON")
    sub = ap.add_subparsers(dest="command", required=True, metavar="command")
    for fn in actions.ACTIONS:
        name = fn.__name__.replace("_", "-")
        doc = inspect.getdoc(fn) or ""
        p = sub.add_parser(name, aliases=ALIASES.get(name, []), help=doc.split("\n\n")[0].replace("\n", " "),
                           description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
        hints, helps = typing.get_type_hints(fn), _param_help(fn)
        for pname, prm in inspect.signature(fn).parameters.items():
            tp, flag = _base(hints[pname]), "--" + pname.replace("_", "-")
            is_list = typing.get_origin(tp) is list
            item = typing.get_args(tp)[0] if is_list else tp
            required = prm.default is inspect.Parameter.empty
            h = helps.get(pname)
            if required:
                p.add_argument(pname, nargs="+" if is_list else None, type=item, help=h)
            elif tp is bool:
                if prm.default:
                    p.add_argument("--no-" + pname.replace("_", "-"), dest=pname, action="store_false", help=h)
                else:
                    p.add_argument(flag, dest=pname, action="store_true", help=h)
            else:
                p.add_argument(flag, dest=pname, type=_comma(item) if is_list else item, default=prm.default,
                               help=(h or "") + (f" (default: {','.join(map(str, prm.default))})" if is_list and prm.default
                                                 else f" (default: {prm.default})" if prm.default is not None else ""))
        p.set_defaults(_fn=fn)
    return ap


def _human(x, pad: str = "") -> None:
    if isinstance(x, dict) and "lines" in x and isinstance(x["lines"], list):   # a transcript
        for ln in x["lines"]:
            flags = f"  ({', '.join(ln['flags'])})" if ln.get("flags") else ""
            print(f"{ln['line']:>4}  [{actions._fmt_time(ln['start'])}] {ln['speaker']}: {ln['text']}{flags}")
        return
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, (dict, list)) and v:
                print(f"{pad}{k}:")
                _human(v, pad + "  ")
            else:
                print(f"{pad}{k}: {v}")
    elif isinstance(x, list):
        for v in x:
            if isinstance(v, dict):
                print(pad + "- " + ", ".join(f"{k}={w}" for k, w in v.items() if not isinstance(w, (dict, list)) or w))
            else:
                print(f"{pad}- {v}")
    else:
        print(f"{pad}{x}")


def main(argv=None) -> int:
    args = vars(build_parser().parse_args(argv))
    fn, as_json = args.pop("_fn"), args.pop("json")
    args.pop("command")
    for stream in (sys.stdout, sys.stderr):    # transcripts can be in any language, also through a pipe
        stream.reconfigure(encoding="utf-8")
    actions.progress_out = lambda m: print(m, file=sys.stderr, flush=True)
    try:
        result = fn(**args)
    except actions.ActionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    if as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _human(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
