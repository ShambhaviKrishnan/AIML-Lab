#Smart Light Agent (Simple Reflex Agent)

#Taking inputs
outside_light = input("Is outside light available? (yes/no): ")
human_presence = input("Is a human present in the room? (yes/no): ")

# Decision making by the agent
if human_presence.lower() == "yes" and outside_light.lower() == "no":
    print("Light Status: ON")
else:
    print("Light Status: OFF")
