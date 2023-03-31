import webcolors
from app.utils import catch_exceptions
import logging
from .check_white_background import CheckBackground
logger = logging.getLogger(name=__name__)

class GetColors(object):
    def __init__(self):
        pass

    @catch_exceptions
    def get_exact_color_name(self,colors_list, color_name):
        try:
            for colors in colors_list:
                for color in colors:
                    if color == color_name:
                        return colors[0]
            return 0
        except Exception as e:
            logger.error(e,exc_info=True)

    @catch_exceptions
    def closest_color(self,requested_color):
        try:
            colors = {}
            for key, name in webcolors.css3_hex_to_names.items():                        # css3_hex_to_name gives a dict whose keys are the hexadecimal values of the 147 names CSS3 colors
                red_pixel, green_pixel, blue_pixel = webcolors.hex_to_rgb(key)           # Converts a hexadecimal color to rbg triplet
                red = (red_pixel - requested_color[0]) ** 2                              
                green = (green_pixel - requested_color[1]) ** 2
                blue = (blue_pixel - requested_color[2]) ** 2
                colors[(red + green + blue)] = name                                      # key will be sum of rgb pixels and value will be color name
            return colors[min(colors.keys())]                                            # return the color which have the minimum pixel value
        except Exception as e:
            logger.error(e,exc_info=True)

    @catch_exceptions
    def get_color_name(self,requested_color):
        try:
            if "red" in requested_color.keys():
                red = requested_color["red"]
               
            else:
                red = 0
            if "blue" in requested_color.keys():
                blue = requested_color["blue"]
            else:
                blue = 0
            if "green" in requested_color.keys():
                green = requested_color["green"]
            else:
                green = 0
            closest_name = webcolors.rgb_to_name((red,green,blue))
        except ValueError:
            closest_name = self.closest_color((red,green,blue))
        return closest_name

    @catch_exceptions
    def get_color_names(self,requested_color):
        try:
            request_colors=requested_color[1:len(requested_color)-1]
            request_color=tuple(map(int,request_colors.split(',')))
            red = int(request_color[0])
            green=int(request_color[1])
            blue=int(request_color[2])
            closest_name = webcolors.rgb_to_name((red,green,blue))
        except ValueError:
            closest_name = self.closest_color((red,green,blue))
        return closest_name


    @catch_exceptions
    def get_colors(self,image):
        try:
            colors_list = [['red', 'indianred', 'lightcoral', 'salmon', 'darksalmon', 'lightsalmon', 'crimson', 'firebrick', 'darkred'],['pink', 'lightpink', 'hotpink', 'deeppink', 'mediumvioletred', 'palevioletred'], ['orange', 'coral', 'tomato', 'orangered', 'darkorange'], ['yellow','darkgoldenrod','gold', 'lightyellow', 'lemonchiffon', 'lightgoldenrodyellow', 'papayawhip', 'moccasin', 'peachpuff', 'palegoldenrod', 'khaki', 'darkkhaki'],['purple', 'lavender', 'thistle', 'plum', 'violet', 'orchid', 'fuchsia', 'magenta', 'mediumorchid', 'mediumpurple', 'blueviolet', 'darkviolet', 'darkorchid', 'darkmagenta', 'rebeccapurple', 'indigo', 'mediumslateblue', 'slateblue', 'darkslateblue'],['green', 'greenyellow', 'chartreuse', 'lawngreen', 'lime', 'limegreen', 'palegreen', 'lightgreen', 'mediumspringgreen', 'springgreen', 'mediumseagreen', 'seagreen', 'forestgreen', 'darkgreen', 'yellowgreen', 'olivedrab', 'olive', 'darkolivegreen', 'mediumaquamarine', 'darkseagreen', 'lightseagreen', 'darkcyan', 'teal'],['blue', 'aqua', 'cyan', 'lightcyan', 'paleturquoise', 'aquamarine', 'turquoise', 'mediumturquoise', 'darkturquoise', 'cadetblue', 'steelblue', 'lightsteelblue', 'powderblue', 'lightblue', 'skyblue', 'lightskyblue', 'deepskyblue', 'dodgerblue', 'cornflowerblue', 'royalblue', 'mediumblue', 'darkblue', 'navy', 'midnightblue'], ['brown', 'cornsilk', 'blanchedalmond', 'bisque', 'navajowhite', 'wheat', 'burlywood', 'tan', 'rosybrown', 'sandybrown', 'peru', 'chocolate', 'saddlebrown', 'sienna',], ['maroon'],['beige'],['white','snow','honeydew', 'mintcream', 'azure', 'aliceblue', 'ghostwhite', 'whitesmoke', 'seashell', 'oldlace', 'floralwhite', 'ivory', 'antiquewhite', 'linen', 'lavenderblush', 'mistyrose'], ['grey', 'gainsboro', 'lightgray', 'lightgrey', 'silver', 'darkgray', 'darkgrey','gray', 'dimgray', 'dimgrey', 'lightslategray', 'lightslategrey', 'slategray', 'slategrey'],['black', 'darkslategray', 'darkslategrey']]
            image_colors = []
            most_frequent_colour=CheckBackground.find_most_frequent_colour(image)
            for color in most_frequent_colour:
                color_name = self.get_color_names(color)
                original_color = self.get_exact_color_name(colors_list, color_name)
                image_colors.append(original_color)
            image_colors = list(set(image_colors[:2]))
            return image_colors
        except Exception as e:
            logger.error(e,exc_info=True)
            return []


Colors = GetColors()