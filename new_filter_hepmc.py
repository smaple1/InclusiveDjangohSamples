#!/usr/bin/env python3

# Note to scm: needs serious optimisation, takes way too much memory currently - likely the way I'm doing logging

"""
Repair HepMC ASCII files:
 - convert partons with status==1 -> status==2
 - detect and break parent->child cycles by setting chosen particle's parent to 0
 - log edits that are made
 - Stop after N events, and skip/drop events if cycle search exceeds step limit

Usage:
    ./new_filter_hepmc.py input.hepmc output_cleaned.hepmc [--max-events N] [--max-steps-per-event M] [--drop-bad-events] [--verbose]
"""
import argparse
from collections import defaultdict

#  list of partons to check 
def is_parton(pid: int) -> bool:
    base_partons = set(range(1, 7)) | {21} | set(range(90, 93))  # quarks 1..6, gluon, special codes
    diquarks = {
        1103, 2101, 2103, 2203, 3101, 3103,
        3201, 3203, 3303, 4101, 4103,
        4201, 4203, 4301, 4303, 4403,
        5101, 5103, 5201, 5203, 5301, 5303, 5401, 5403,
        5503,
    }
    return abs(pid) in base_partons or abs(pid) in diquarks


def parse_p_line(line):
    """Parse P-line. Returns dict or None if can't parse."""
    parts = line.strip().split()
    if len(parts) < 3 or parts[0] != "P":
        return None
    try:
        idx = int(parts[1])
    except Exception:
        return None

    parent = 0
    pid = 0
    parent_field = False
    status = None

    if len(parts) >= 4:
        try:
            parent = int(parts[2])
            pid = int(parts[3])
            parent_field = True
        except ValueError:
            try:
                pid = int(parts[2])
                parent = 0
                parent_field = False
            except Exception:
                pid = 0
                parent = 0
    else:
        try:
            pid = int(parts[2])
        except Exception:
            pid = 0

    try:
        status = int(parts[-1])
    except Exception:
        status = None

    return {
        "idx": idx,
        "parent": parent,
        "pid": pid,
        "status": status,
        "parts": parts,
        "parent_field": parent_field,
        "line": line,
    }


def detect_and_break_cycles(event_particles, lines, edits, event_id, max_steps=100000, verbose=False):
    """Detect cycles in parent links. Returns step_count."""
    parent_map = {}
    for idx, info in event_particles.items():
        p = info["parent"]
        parent_map[idx] = p if (p and p in event_particles) else 0

    visited, onstack, parent_trace = set(), set(), {}
    cycles_found = []
    step_count = 0

    def dfs(node):
        nonlocal step_count
        step_count += 1
        if step_count > max_steps:
            raise RuntimeError(
                f"Event {event_id}: exceeded max DFS steps ({max_steps}), skipping cycle detection"
            )
        visited.add(node)
        onstack.add(node)
        nxt = parent_map.get(node, 0)
        if nxt and nxt in event_particles:
            if nxt not in visited:
                parent_trace[nxt] = node
                dfs(nxt)
            elif nxt in onstack:
                cycle = [nxt]
                cur = node
                while cur != nxt and cur in parent_trace:
                    cycle.append(cur)
                    cur = parent_trace[cur]
                if cur == nxt:
                    uniq = []
                    for x in reversed(cycle):
                        if x not in uniq:
                            uniq.append(x)
                    cycles_found.append(uniq)
        onstack.remove(node)

    for n in list(event_particles.keys()):
        if n not in visited:
            parent_trace.clear()
            dfs(n)

    handled_nodes = set()
    for cycle in cycles_found:
        if any(n in handled_nodes for n in cycle):
            continue
        parton_candidates = [n for n in cycle if is_parton(event_particles[n]["pid"])]
        chosen = max(parton_candidates) if parton_candidates else max(cycle)
        info = event_particles[chosen]
        old_parent = info["parent"]
        info["parent"] = 0
        line_idx = info["line_idx"]
        if info["parent_field"]:
            parts = info["parts"]
            if len(parts) >= 3:
                parts[2] = "0"
                lines[line_idx] = " ".join(parts) + "\n"
            else:
                edits.append(f"Event {event_id}: chosen {chosen} had no parent field to set to 0; left unchanged.")
        else:
            edits.append(f"Event {event_id}: chosen {chosen} had no parent field; logically set to 0.")
        edits.append(f"Event {event_id}: cycle detected {cycle} -> reset parent of {chosen} (was {old_parent}) to 0.")
        handled_nodes.update(cycle)

    if verbose:
        edits.append(f"Event {event_id}: cycle detection used {step_count} steps")
    return step_count


def repair_hepmc_file(input_path, output_path, max_events=None, max_steps_per_event=100000, drop_bad_events=False, verbose=False, progress_every=10000):
    with open(input_path, "r") as f:
        lines = f.readlines()

    current_event = None
    events_particles = defaultdict(dict)
    event_ranges = {}  # event_id -> (start_line, end_line)
    all_edits = defaultdict(list)
    start_line = None

    for li, line in enumerate(lines):
        if line.startswith("E "):
            if start_line is not None and current_event is not None:
                event_ranges[current_event] = (start_line, li)
            start_line = li
            parts = line.split()
            if len(parts) >= 2:
                try:
                    current_event = int(parts[1])
                    if progress_every and current_event % progress_every == 0:
                        print(f"[Progress] processed {current_event} events")
                    if max_events and current_event > max_events:
                        break
                except Exception:
                    current_event = None
            else:
                current_event = None
            continue

        if line.startswith("P "):
            info = parse_p_line(line)
            if info is None:
                continue
            info["line_idx"] = li
            info["event"] = current_event
            events_particles[current_event][info["idx"]] = info

    if start_line is not None and current_event is not None:
        event_ranges[current_event] = (start_line, len(lines))

    events_to_drop = set()

    for evt, part_dict in events_particles.items():
        edits = []
        for idx, info in list(part_dict.items()):
            if info["status"] == 1 and is_parton(info["pid"]):
                parts = info["parts"]
                parts[-1] = "2"
                info["parts"] = parts
                info["status"] = 2
                lines[info["line_idx"]] = " ".join(parts) + "\n"
                edits.append(f"Particle {idx} (PDG {info['pid']}) had status=1 -> set status=2")
        try:
            detect_and_break_cycles(part_dict, lines, edits, evt,
                                    max_steps=max_steps_per_event,
                                    verbose=verbose)
        except RuntimeError as e:
            if drop_bad_events:
                events_to_drop.add(evt)
                edits.append(f"{e} -> dropping event")
            else:
                edits.append(str(e))
        if edits:
            all_edits[evt].extend(edits)

    if events_to_drop:
        keep_lines = []
        drop_ranges = [event_ranges[evt] for evt in events_to_drop if evt in event_ranges]
        drop_set = set()
        for start, end in drop_ranges:
            drop_set.update(range(start, end))

        for li, line in enumerate(lines):
            if li not in drop_set:
                keep_lines.append(line)
        lines = keep_lines
    # if events_to_drop:
        # keep_lines = []
        # for li, line in enumerate(lines):
            # keep = True
            # if line.startswith("E "):
                # parts = line.split()
                # if len(parts) >= 2:
                    # try:
                        # evt_id = int(parts[1])
                        # if evt_id in events_to_drop:
                            # keep = False
                    # except Exception:
                        # pass
            # if keep:
                # keep_lines.append(line)
        # lines = keep_lines

    with open(output_path, "w") as f:
        f.writelines(lines)

    if all_edits:
        print("=== Repair summary ===")
        for evt in sorted(all_edits.keys()):
            print(f"Event {evt}:")
            for e in all_edits[evt]:
                print("  -", e)
    else:
        print("No repairs needed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repair HepMC ASCII (break loops, fix parton status).")
    parser.add_argument("input", help="Input HepMC ASCII file")
    parser.add_argument("output", help="Output cleaned HepMC file")
    parser.add_argument("--max-events", type=int, help="Stop after processing this many events")
    parser.add_argument("--max-steps-per-event", type=int, default=100000,
                        help="Abort cycle detection if DFS exceeds this many steps")
    parser.add_argument("--drop-bad-events", action="store_true",
                        help="Drop entire events that exceed max DFS steps")
    parser.add_argument("--verbose", action="store_true", help="Verbose output (print step counts)")
    args = parser.parse_args()

    repair_hepmc_file(args.input, args.output,
                      max_events=args.max_events,
                      max_steps_per_event=args.max_steps_per_event,
                      drop_bad_events=args.drop_bad_events,
                      verbose=args.verbose)
    print(f"[Done] Cleaned HepMC written to {args.output}")
