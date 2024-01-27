import numpy as np
import tensorflow as tf

classes = [
    "air hockey",
    "ampute football",
    "archery",
    "arm wrestling",
    "axe throwing",
    "balance beam",
    "barell racing",
    "baseball",
    "basketball",
    "baton twirling",
    "bike polo",
    "billiards",
    "bmx",
    "bobsled",
    "bowling",
    "boxing",
    "bull riding",
    "bungee jumping",
    "canoe slamon",
    "cheerleading",
    "chess",    
    "chuckwagon racing",
    "cricket",
    "croquet",
    "curling",
    "dance sports",
    "disc golf",
    "esport",
    "fencing",
    "field hockey",
    "figure skating men",
    "figure skating pairs",
    "figure skating women",
    "fly fishing",
    "football (UK)",
    "formula 1 racing",
    "frisbee",
    "gaga",
    "giant slalom",
    "golf",
    "hammer throw",
    "hang gliding",
    "harness racing",
    "high jump",
    "hockey",
    "horse jumping",
    "horse racing",
    "horseshoe pitching",
    "hurdles",
    "hydroplane racing",
    "ice climbing",
    "ice yachting",
    "jai alai",
    "javelin",
    "jousting",
    "judo",
    "kayaking",
    "lacrosse",
    "log rolling",
    "luge",
    "motorcycle racing",
    "mushing",
    "nascar racing",
    "olympic wrestling",
    "paintball",
    "parallel bar",
    "parkour",
    "pole climbing",
    "pole dancing",
    "pole vault",
    "polo",
    "pommel horse",
    "rings",
    "rock climbing",
    "roller derby",
    "rollerblade racing",
    "rowing",
    "rugby",
    "sailboat racing",
    "sandboarding",
    "scuba diving",
    "shot put",
    "shuffleboard",
    "sidecar racing",
    "ski jumping",
    "sky surfing",
    "skydiving",
    "snow boarding",
    "snowmobile racing",
    "soccer",   
    "speed skating",
    "steer wrestling",
    "sumo wrestling",
    "surfing",
    "swimming",
    "table tennis",
    "tennis",
    "track and field",
    "track bicycle",
    "trapeze",
    "tug of war",
    "ultimate",
    "uneven bars",
    "volleyball",
    "water cycling",
    "water polo",
    "weightlifting",
    "wheelchair basketball",
    "wheelchair racing",
    "wingsuit flying",
]


def predict_label(img, model):
    resized_img = tf.image.resize(img, (224, 224)).numpy().astype(int)
    exp_img = np.expand_dims(resized_img, 0)
    y_prob = model.predict(exp_img)
    if y_prob.max(axis=-1) < 0.5:
        return "Cannot predict. Please input appropriate image."
    else:
        y_classes = y_prob.argmax(axis=-1)
        label = classes[y_classes[0]]
        return "Predicted Sport: " + label.capitalize()
    

