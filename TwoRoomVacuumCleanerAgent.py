# Two-Room Vacuum Cleaner Agent Simulation

# Initial state of rooms
rooms = {
    "A": "Dirty",
    "B": "Dirty"
}

# Agent starting position
current_room = "A"

print("Initial State:", rooms)

while rooms["A"] == "Dirty" or rooms["B"] == "Dirty":

    print("\nAgent is in Room", current_room)

    if rooms[current_room] == "Dirty":
        print("Room", current_room, "is Dirty. Cleaning...")
        rooms[current_room] = "Clean"

    else:
        print("Room", current_room, "is already Clean. Moving to other room.")

        if current_room == "A":
            current_room = "B"
        else:
            current_room = "A"

    print("Current State:", rooms)

print("\nBoth rooms are clean. Task completed.")
