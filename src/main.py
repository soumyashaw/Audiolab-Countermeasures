from simple_term_menu import TerminalMenu

def main():
    # Define the menu options
    menu_options = [
        "Calculate Average STI",
        "Trial 2",
        "Exit",
    ]

    # Create a TerminalMenu instance
    terminal_menu = TerminalMenu(menu_options, title="Main Menu", clear_screen=True)

    while True:
        # Show the menu and get the selected option
        selected_option_index = terminal_menu.show()

        # Perform actions based on the selected option
        if selected_option_index == 0:
            print("Hello, World!")

        elif selected_option_index == 1:
            from datetime import datetime
            print(f"Current date and time: {datetime.now()}")
            
        elif selected_option_index == 2:
            print("Exiting the program. Goodbye!")
            break

        # Pause to allow the user to read the output
        input("\nPress Enter to return to the menu...")

if __name__ == "__main__":
    main()