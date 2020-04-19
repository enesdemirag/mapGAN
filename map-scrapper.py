from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
import time

CHROMEDRIVER_PATH = '/usr/local/bin/chromedriver'
driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH)
driver.get("https://www.redblobgames.com/maps/mapgen2/embed.html#seed=1&size=small&noisy-fills=true&noisy-edges=true&lighting=true")
driver.find_element_by_xpath('//*[@id="icons"]').click()

i = 0
while True:
    element = driver.find_element_by_id("map")
    location = element.location
    size = element.size
    
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
    im.save('dataset/' + str(i) + '.png')
    # Next
    driver.find_elements_by_xpath('//*[@id="ui"]/div[2]/div[1]/button[2]')[0].click()
    time.sleep(0.5)
    i = i + 1

driver.close()
