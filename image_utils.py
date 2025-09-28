import os
import PIL

import cv2
from PIL import Image
import google.generativeai as genai

image_list = []

def save_clipped_person(image, bbox, person_id, frame_index):
    save_path = "suspect_images/"
    """
    Clips and saves a bounding box region from the image.

    image: numpy array (BGR frame from cv2)
    bbox: (x1, y1, x2, y2) coordinates of the bounding box
    save_path: directory to save images
    person_id: unique id for the person
    """
    x1, y1, x2, y2 = bbox

    # clip coordinates to image boundaries
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # crop
    cropped = image[y1:y2, x1:x2]

    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cropped_rgb)

    image_list.append(pil_img)
    generate_summary(image_list)


#needs a list of strings and the strings are image names like image.jpg
def generate_summary(image_list):

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    prompt = """
        I need you to give a description of the person depicted in the images I attached. 
        It should be a quality description that the police could use to identify the suspect
        Here is an example of what it should look like, you will follow this exact format, for each row it will say what your answer should look like
        Sex: MALE OR FEMALE ex. Male
        Race: Figure out what the persons race is based on skin color and appearance ex. White
        Height: Guess a height range based how the person appears compared to surroungding objects ex. 5'4-5'7
        Weight: Guess the weight based on height and appearance, ex. 180-200lbs
        Build: Fat or Slim or Muscular or Average
        Hair Color: Figure out the hair color ex. Blonde
        Hair Style: Figure out the hair style ex. Short and Spiky
        Eye Color: Figure out eye color ex. Hazel
        Nose Shape: Long or Wide or Flat or Average ex. Long
        Facial Hair: Does the person have a beard or mustache or clean ex. Beard and Mustache
        Tatoos: Does the person have any tatoos ex. Large tatoo on the right thigh
        Hat/Coverings: Is the person wearing a hat or facial covering ex. Blue Baseball Cap and a Black Ski Mask
        Coat: Is the person wearing a coat or jacket ex. Grey Winter Coat
        Shirt: Describe the persons shirt ex. Light Blue Shirt
        Shoes: Describe the persons shoes ex. Black Boots
        Accessories: Describe any accessories ex. Blue Scarf and a Black Gloves
        Jewelry: Decribe any jewelery ex. Silver Watch
        General Appearance: Neat or Sloppy ex. Well Dressed
        Other Comments: Here you need to put the most identifiable features of the person, this is for anything important that you did not already mention

        You need to be very sure for each category if you cannot figure out a category then just put UI (stands for unidentified)
        ex. If you can't figure out the eye color just put UI, I want you to try and identify things but you need to BE SURE do not make random guesses and lie I would rather have u put UI than lie
        Be flexible with you descriptions, for example if you see something gold around the persons wrist but you don't know if its a watch or bracelet DO NOT SAY UI, you would just say Jewelery: Gold item around wrist
        Be helpful and intelligent, make sure to use ALL of the images and if something is only in one image but not the other 4 STILL MENTION IT in the the decription, some things may only be visible in 1-2 images        
        """

    model = genai.GenerativeModel('gemini-2.5-flash')

    response = model.generate_content([prompt] + image_list)

    print(response.text)