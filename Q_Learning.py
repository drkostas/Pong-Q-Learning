"""
usage: python Q_Learning.py q a e n c f
"""
import sys

def main(args):
    """ Main function"""
    print("------ Initializing ------")
    # --- Args Loading and Error Checking --- #
    if len(args) == 6:
        # Load the arguments
        q, a, e, n, c, f = args[:5]
        # TODO:Cast the arguments to the correct type
    else:
        # If more or less arguments are given, print an error message
        raise Exception("Invalid number of arguments")
    
    # Call some functions


if __name__ == "__main__":
    main(sys.argv[1:])