import colorsys


def hex_to_hsl(hex_color):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r /= 255.0
    g /= 255.0
    b /= 255.0

    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)

    hue = round(hue * 360)
    saturation = round(saturation * 100)
    lightness = round(lightness * 100)

    return hue, saturation, lightness
