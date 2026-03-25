"""EB-Alfred action space definitions.

Reference: EmbodiedBench/embodiedbench/envs/eb_alfred/utils.py
           EmbodiedBench/embodiedbench/envs/eb_alfred/EBAlfEnv.py
"""
from __future__ import annotations

# All objects in EB-Alfred (alfred_objs from EmbodiedBench)
EBALFRED_OBJECTS = [
    "Cart", "Potato", "Faucet", "Ottoman", "CoffeeMachine", "Candle", "CD",
    "Pan", "Watch", "HandTowel", "SprayBottle", "BaseballBat", "CellPhone",
    "Kettle", "Mug", "StoveBurner", "Bowl", "Toilet", "DiningTable", "Spoon",
    "TissueBox", "Shelf", "Apple", "TennisRacket", "SoapBar", "Cloth",
    "Plunger", "FloorLamp", "ToiletPaperHanger", "CoffeeTable", "Spatula",
    "Plate", "Bed", "Glassbottle", "Knife", "Tomato", "ButterKnife",
    "Dresser", "Microwave", "CounterTop", "GarbageCan", "WateringCan", "Vase",
    "ArmChair", "Safe", "KeyChain", "Pot", "Pen", "Cabinet", "Desk",
    "Newspaper", "Drawer", "Sofa", "Bread", "Book", "Lettuce", "CreditCard",
    "AlarmClock", "ToiletPaper", "SideTable", "Fork", "Box", "Egg",
    "DeskLamp", "Ladle", "WineBottle", "Pencil", "Laptop", "RemoteControl",
    "BasketBall", "DishSponge", "Cup", "SaltShaker", "PepperShaker", "Pillow",
    "Bathtub", "SoapBottle", "Statue", "Fridge", "Sink",
]

# Pickupable objects (alfred_pick_obj)
EBALFRED_PICKUPABLE = [
    "KeyChain", "Potato", "Pot", "Pen", "Candle", "CD", "Pan", "Watch",
    "Newspaper", "HandTowel", "SprayBottle", "BaseballBat", "Bread",
    "CellPhone", "Book", "Lettuce", "CreditCard", "Mug", "AlarmClock",
    "Kettle", "ToiletPaper", "Bowl", "Fork", "Box", "Egg", "Spoon",
    "TissueBox", "Apple", "TennisRacket", "Ladle", "WineBottle", "Cloth",
    "Plunger", "SoapBar", "Pencil", "Laptop", "RemoteControl", "BasketBall",
    "DishSponge", "Cup", "Spatula", "SaltShaker", "Plate", "PepperShaker",
    "Pillow", "Glassbottle", "SoapBottle", "Knife", "Statue", "Tomato",
    "ButterKnife", "WateringCan", "Vase",
]

# Openable objects (alfred_open_obj)
EBALFRED_OPENABLE = [
    "Safe", "Laptop", "Fridge", "Box", "Microwave", "Cabinet", "Drawer",
]

# Sliceable objects (alfred_slice_obj)
EBALFRED_SLICEABLE = [
    "Potato", "Lettuce", "Tomato", "Apple", "Bread",
]

# Toggleable objects (alfred_toggle_obj)
EBALFRED_TOGGLEABLE = [
    "Microwave", "DeskLamp", "FloorLamp", "Faucet",
]

# Receptacles (alfred_recep)
EBALFRED_RECEPTACLES = [
    "ArmChair", "Safe", "Cart", "Ottoman", "Pot", "CoffeeMachine", "Desk",
    "Cabinet", "Pan", "Drawer", "Sofa", "Mug", "StoveBurner", "SideTable",
    "Toilet", "Bowl", "Box", "DiningTable", "Shelf", "ToiletPaperHanger",
    "CoffeeTable", "Cup", "Plate", "Bathtub", "Bed", "Dresser", "Fridge",
    "Microwave", "CounterTop", "Sink", "GarbageCan",
]


def get_global_action_space() -> list[str]:
    """Generate the full EB-Alfred action space (static, ~133 actions).

    Reference: EmbodiedBench/embodiedbench/envs/eb_alfred/EBAlfEnv.py
    """
    actions = []
    # Find actions for all objects
    actions.extend(f"find a {obj}" for obj in EBALFRED_OBJECTS)
    # Pickup actions
    actions.extend(f"pick up the {obj}" for obj in EBALFRED_PICKUPABLE)
    # Put / drop
    actions.extend(["put down the object in hand", "drop the object in hand"])
    # Open / close
    for obj in EBALFRED_OPENABLE:
        actions.extend([f"open the {obj}", f"close the {obj}"])
    # Toggle on/off
    for obj in EBALFRED_TOGGLEABLE:
        actions.extend([f"turn on the {obj}", f"turn off the {obj}"])
    # Slice
    actions.extend(f"slice the {obj}" for obj in EBALFRED_SLICEABLE)
    return actions
