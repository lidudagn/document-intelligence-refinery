#!/usr/bin/env python3
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Document Intelligence Refinery CLI",
        usage="python refinery_cli.py <command> [<args>...]"
    )
    
    parser.add_argument(
        "command", 
        choices=["triage", "extract", "index", "query", "batch"], 
        help="Command to run: triage, extract, index, query, or batch"
    )
    
    args, unknown = parser.parse_known_args()
    
    cmd_map = {
        "triage": "run_triage.py",
        "extract": "run_extraction.py",
        "index": "run_indexer.py",
        "query": "run_query_agent.py",
        "batch": "run_full_batch.py"
    }
    
    script = cmd_map.get(args.command)
    if not script:
        parser.print_help()
        sys.exit(1)
        
    cmd = [sys.executable, script] + unknown
    
    try:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)

if __name__ == "__main__":
    main()
