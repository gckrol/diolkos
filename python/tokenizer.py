"""
From lm.rs.

MIT
Copyright (c) 2024 Samuel
Copyright (c) 2025 Krol Inventions B.V.
"""
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-id", type=str, required=True, help="model id for the corresponding tokenizer")
    parser.add_argument("-t", "--tokenizer-type", type=str, required=True, choices=['LLAMA', 'GEMMA', 'PHI'], help="type of tokenizer (GEMMA/LLAMA/PHI)")
    args = parser.parse_args()

    if args.tokenizer_type == "GEMMA":        
        from tok.gemma import Tokenizer
    elif args.tokenizer_type == "LLAMA":
        from tok.llama import Tokenizer
    elif args.tokenizer_type == "PHI":
        from tok.phi import Tokenizer
    
    t = Tokenizer(args.model_id)
    t.export()
