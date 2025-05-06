import torch
from collections import OrderedDict
import argparse

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Remove dense_head weights from model checkpoint')
    
    # Add command line arguments
    parser.add_argument('--input_path', type=str, required=True, 
                        help='Input model checkpoint path')
    parser.add_argument('--output_path', type=str, required=True, 
                        help='Output model checkpoint path')
    parser.add_argument('--remove_key', type=str, default='dense_head',
                        help='Key prefix to remove, default is dense_head')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Load model checkpoint
    print(f"Loading checkpoint from: {args.input_path}")
    ckpt = torch.load(args.input_path)
    
    # Create new model state dictionary
    new_ckpt = OrderedDict()
    remove_count = 0
    keep_count = 0
    
    # Iterate and filter weights
    for key, val in ckpt['model_state'].items():
        if args.remove_key not in key:
            new_ckpt[key] = val
            keep_count += 1
        else:
            remove_count += 1
    
    # Save new model checkpoint
    save_ckpt = {}
    save_ckpt['model_state'] = new_ckpt
    
    print(f"Removed {remove_count} weights, kept {keep_count} weights")
    print(f"Saving checkpoint to: {args.output_path}")
    torch.save(save_ckpt, args.output_path)
    print("Save completed")

if __name__ == "__main__":
    main()