import xml.etree.ElementTree as ET

# Define the label values
label_values = ["door", "window", "sword", "treasure", "bush", "tree", "pillow", "book", "box", "sphere", "bird", "barrel", "painting", "stick", "moon", "lamp", "instrument", "table", "chair", "skull", "telescope", "globe", "mirror", "lamp", "vase", "hat", "moon", "pillar", "crate", "coin"]

# Create the root element
root = ET.Element("View")

# Add the Header element
header = ET.SubElement(root, "Header")
header.set("value", "Select label and click the image to start")

# Add the Image element
image = ET.SubElement(root, "Image")
image.set("name", "image")
image.set("value", "$img")
image.set("zoom", "true")

# Add the PolygonLabels element
polygon_labels = ET.SubElement(root, "PolygonLabels")
polygon_labels.set("name", "poly_label")
polygon_labels.set("toName", "image")
polygon_labels.set("strokeWidth", "3")
polygon_labels.set("pointSize", "small")
polygon_labels.set("opacity", "0.9")

# Add the Label elements
for label_value in label_values:
    label = ET.SubElement(polygon_labels, "Label")
    label.set("value", label_value)
    label.set("background", "#FFA39E")  # You can change the background color here

# # Add the RectangleLabels element
# rectangle_labels = ET.SubElement(root, "RectangleLabels")
# rectangle_labels.set("name", "rect_label")
# rectangle_labels.set("toName", "image")

# # Add the Label elements
# for label_value in label_values:
#     label = ET.SubElement(rectangle_labels, "Label")
#     label.set("value", label_value)
#     label.set("background", "#FFA39E")  # You can change the background color here


# Create the tree and write it to a file
tree = ET.ElementTree(root)
tree.write("datasets/label-studio-scripts/interface_definition.xml")