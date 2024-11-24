import pexpect

for i in range(8):
    print(f"Starting Cat {i+1}")
    # Start the program
    child = pexpect.spawn(f"python src/main.py --target_dir='/Users/soumyashaw/Bark/' --reference_dir='/Users/soumyashaw/Bark/cat{i+1}'")

    # Debugging: Log the program output to the console
    child.logfile = open("debug.log", "wb")

    # Wait for the menu to appear and simulate a selection
    # For example, send "2" to select the second option and press Enter
    child.expect("Calculate Average STI")
    child.expect("Utility Based Data Augmentation")
    child.expect("Specific Perturbation Based Data Augmentation")
    child.expect("Exit")
    child.send("\x1b[B")
    child.send("\x1b[B")
    child.sendline("")

    if i == 0:
        child.sendline("")
    elif i == 1:
        child.send("\x1b[B")
        child.sendline("")
    elif i == 2:
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.sendline("")
    elif i == 3:
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.sendline("")
    elif i == 4:
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.sendline("")
    elif i == 5:
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.sendline("")
    elif i == 6:
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.sendline("")
    elif i == 7:
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.send("\x1b[B")
        child.sendline("")

    # Wait for the next menu or prompt
    child.expect(pexpect.EOF)  # Wait until the program exits

    # Clean up
    child.close()

    print(f"Cat{i+1} Done")