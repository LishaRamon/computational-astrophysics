"""Example: 
Write a function that given the height of a ball determines the time it takes to hit the ground. 
h = 1/2 g t2. Allow g to be a keyword so you can use this code on other planets."""

# sys.argv example: Run by itself
"""import math
import sys

def time_to_floor(height, g=9.81):
    return math.sqrt(2 * height / g)

# Check # of arg
if len(sys.argv) < 2:
    print("Usage: codeastro.py <height> [gravity]")
    print("Example: python codeastro.py 10")
    print("Example: python codeastro.py 10 1.62")
    sys.exit(1)

# Get height from command line
height = float(sys.argv[1])

# Get gravity (optional, defaults to Earth)
if len(sys.argv) >= 3:
    gravity = float(sys.argv[2])
else:
    gravity = 9.81  #Earth's gravity

# Calculate and display result
time = time_to_floor(height, gravity)
print(f"A ball dropped from {height}m takes {time:.2f} seconds to hit the floor")"""





# Gemini and Claude AI Used to assist learning
# Argparse example: Run block by itself
import argparse
import math

def time_to_floor(height, g=9.81):
    return math.sqrt(2 * height / g)

# argument parser here
parser = argparse.ArgumentParser(description='Calculate time for any ball to hit the floor')

#arguments here
parser.add_argument('height', type=float, help='Height in meters')
parser.add_argument('-g', '--gravity', type=float, default=9.81, # can use 'g' by itself for short but then change args.gravity to args.g '
                    help='Gravity default (Earth)')

# parse the arguments
args = parser.parse_args()

# show result
time = time_to_floor(args.height, args.gravity)
print(f"A ball dropped from {args.height}m takes {time:.2f} seconds to hit the ground")

# examples of different gravities to use
if args.gravity == 9.81:
    print("(Using Earth gravity)")
elif args.gravity == 1.62:
    print("(Using Moon gravity)")
elif args.gravity == 3.71:
    print("(Using Mars gravity)")