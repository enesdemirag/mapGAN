# Import packages
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
import time
import os

# Browser driver location
CHROMEDRIVER_PATH = '/usr/local/bin/chromedriver'
driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH)

# Define parameters
map_resolution = "huge"
show_icons = False
seed = 0

# Open mapgen2
driver.get("https://www.redblobgames.com/maps/mapgen2/embed.html#seed=" + str(seed) + "&size=" + map_resolution + "&noisy-fills=true&noisy-edges=true&lighting=true")

# Configure
driver.find_element_by_xpath('//*[@id="size-' + map_resolution + '"]').click()
driver.find_element_by_id('seed').clear()
driver.find_element_by_id('seed').send_keys(str(seed))
if not show_icons:
    driver.find_element_by_xpath('//*[@id="icons"]').click()

# Wait for refresh
time.sleep(1)

# Loop
while True:
    # Find map
    element = driver.find_element_by_id("map")
    location = element.location
    size = element.size

    # Get size
    x = location['x']
    y = location['y']
    width = location['x'] + size['width']
    height = location['y'] + size['height']

    # Screenshot
    driver.save_screenshot("temp.png")

    # Crop
    im = Image.open("temp.png")
    im = im.crop((int(x), int(y), int(width), int(height)))

    # Resize
    im = im.resize((512, 512))

    # Save
    im.save('dataset/' + str(seed) + '.png')

    # Next seed
    driver.find_elements_by_xpath('//*[@id="ui"]/div[2]/div[1]/button[2]')[0].click()
    
    # Wait for refresh
    time.sleep(1)
    seed = seed + 1

driver.close()
os.remove("temp.png")
