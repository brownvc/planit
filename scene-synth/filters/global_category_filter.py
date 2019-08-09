from data import ObjectCategories

"""
Just a list of categories that should be filtered out
Shared by all room specific filters
"""
class GlobalCategoryFilter():

    category_map = ObjectCategories()

    def get_filter():

        door_window = GlobalCategoryFilter.category_map.all_arch_categories()
        
        second_tier = ["table_lamp",
                       "chandelier",
                       "guitar",
                       "amplifier",
                       "keyboard",
                       "drumset",
                       "microphone",
                       "accordion",
                       "toy",
                       "xbox",
                       "playstation",
                       "fishbowl",
                       "chessboard",
                       "iron",
                       "helmet",
                       "telephone",
                       "stationary_container",
                       "ceiling_fan",
                       "bottle",
                       "fruit_bowl",
                       "glass",
                       "knife_rack",
                       "plates",
                       "books",
                       "book",
                       "television",
                       "wood_board",
                       "switch",
                       "pillow",
                       "laptop",
                       "clock",
                       "helmet",
                       "bottle",
                       "trinket",
                       "glass",
                       "range_hood",
                       "candle",
                       "soap_dish"]
        
        wall_objects = ["wall_lamp",
                        "mirror",
                        "curtain",
                        "blind"]

        unimportant = ["toy",
                       "fish_tank",
                       "tricycle",
                       "vacuum_cleaner",
                       "weight_scale",
                       "heater",
                       "picture_frame",
                       "beer",
                       "shoes",
                       "weight_scale",
                       "decoration",
                       "ladder",
                       "tripod",
                       "air_conditioner",
                       "cart",
                       "fireplace_tools",
                       "vase"]

        inhabitants = ["person",
                       "cat",
                       "bird",
                       "dog",
                       "pet"]

        special_filter = ["rug",]

        filtered = second_tier + unimportant + inhabitants + special_filter + wall_objects

        unwanted_complex_structure = ["partition",
                                      "column",
                                      "arch",
                                      "stairs"]
        set_items = ["chair_set",
                     "stereo_set",
                     "table_and_chair",
                     "slot_machine_and_chair",
                     "kitchen_set",
                     "double_desk",
                     "double_desk_with_chairs",
                     "dressing_table_with_stool",
                     "kitchen_island_with_range_hood_and_table"]

        outdoor = ["lawn_mower",
                   "car",
                   "motorcycle",
                   "bicycle",
                   "garage_door",
                   "outdoor_seating",
                   "fence"]

        rejected = unwanted_complex_structure + set_items + outdoor
        
        return filtered, rejected, door_window

    def get_filter_latent():
        
        door_window = GlobalCategoryFilter.category_map.all_arch_categories()

        second_tier_include = ["table_lamp",
                               "television",
                               "picture_frame",
                               "books",
                               "book",
                               "laptop",
                               "floor_lamp",
                               "vase",
                               "plant",
                               "console",
                               "stereo_set",
                               "toy",
                               "fish_tank",
                               "cup",
                               "glass",
                               "fruit_bowl",
                               "bottle",
                               "fishbowl",
                               "pillow",
                               ]
                        
        second_tier = ["chandelier",
                       "guitar",
                       "amplifier",
                       "keyboard",
                       "drumset",
                       "microphone",
                       "accordion",
                       "chessboard",
                       "iron",
                       "helmet",
                       "stationary_container",
                       "ceiling_fan",
                       "knife_rack",
                       "plates",
                       "wood_board",
                       "switch",
                       "clock",
                       "helmet",
                       "trinket",
                       "range_hood",
                       "candle",
                       "soap_dish"]
        
        wall_objects = ["wall_lamp",
                        "mirror",
                        "curtain",
                        "wall_shelf",
                        "blinds",
                        "blind"]

        unimportant = ["tricycle",
                       "fish_tank",
                       "vacuum_cleaner",
                       "weight_scale",
                       "heater",
                       "picture_frame",
                       "beer",
                       "shoes",
                       "weight_scale",
                       "decoration",
                       "ladder",
                       "tripod",
                       "air_conditioner",
                       "cart",
                       "fireplace_tools",
                       "ironing_board",
                       ]

        inhabitants = ["person",
                       "cat",
                       "bird",
                       "dog",
                       "pet"]

        special_filter = ["rug",]

        filtered = second_tier + unimportant + inhabitants + special_filter + wall_objects

        unwanted_complex_structure = ["partition",
                                      "column",
                                      "arch",
                                      "stairs"]
        set_items = ["chair_set",
                     "stereo_set",
                     "table_and_chair",
                     "slot_machine_and_chair",
                     "kitchen_set",
                     "double_desk",
                     "double_desk_with_chairs",
                     "desk_with_shelves",
                     "dressing_table_with_stool",
                     "armchair_with_ottoman",
                     "kitchen_island_with_range_hood_and_table"]

        outdoor = ["lawn_mower",
                   "car",
                   "motorcycle",
                   "bicycle",
                   "garage_door",
                   "outdoor_seating",
                   "fence"]

        rejected = unwanted_complex_structure + set_items + outdoor

        return filtered, rejected, door_window, second_tier_include



                          



