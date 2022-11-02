"""
usage: python3 runEpisode.py q f
"""
import sys

def main(args):
    """ Main function"""
    print("------ Initializing ------")
    # --- Args Loading and Error Checking --- #
    if len(args) == 2:
        # Load the arguments
        q, f = args[:2]
        # TODO:Cast the arguments to the correct type
    else:
        # If more or less arguments are given, print an error message
        raise Exception("Invalid number of arguments")
    
    # Call some functions


if __name__ == "__main__":
    main(sys.argv[1:])