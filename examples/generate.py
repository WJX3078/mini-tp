"""Example: greedy generation with mini-TP.

Single GPU:      python examples/generate.py
TP=2 multi GPU:  torchrun --standalone --nproc-per-node=2 examples/generate.py
"""

from minitp.generate import main

if __name__ == "__main__":
    main()
