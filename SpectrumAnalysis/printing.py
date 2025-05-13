from astropy.io import fits
def print_help_message():
    print_color_message(f"Press the index number to choose the file to read into the hdu", color_code=33)# yellow
    print_color_message(f"Press 'e' to end the addition of HDU", color_code=33)
    print_color_message(f"Press 'n' to go to the next zip file", color_code=33)
    print_color_message(f"Press 'p' to go to the previous zip file", color_code=33)

def print_color_message(text, color_code):
    print(f"\033[{color_code}m{text}\033[0m", flush=True)