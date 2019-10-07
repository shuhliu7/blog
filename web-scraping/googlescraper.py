from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep
"""
use two words to scrape questions from google people also ask

author: shuhao Liu / harrymj19961007@gmail.com
"""

def scraper(question_list,deepth,sleep_time):
    '''
    question_list: string question list   example:['apple stock ?']
    deepth: int  the deepth determines the amount of questions generated
    sleep_time: float  the wait time for the driver to refresh the page
    '''
    text_list = []
    for question in question_list:
        try:
            output = single_question_scraper(question,deepth,sleep_time)
            for i in output:
                text_list.append(i.text)
        except:
            print('something wrong with '+ question)
            pass

        driver.find_element_by_name('q').clear()
    print(text_list)

def single_question_scraper(question,deepth,sleep_time):
    '''
     question : is the two words and a '?' (question mark)
     deepth : the deepth of information we want 

    '''
    
    if deepth == 0 :
        search =driver.find_element_by_name('q')
        search.send_keys(question)
        search.send_keys(Keys.RETURN)
        sleep(3)
        output = driver.find_elements_by_xpath('//div[@class="match-mod-horizontal-padding hide-focus-ring cbphWd"]')
        return(output)
    if deepth != 0:
        output = single_question_scraper(question,deepth-1,sleep_time)
        length = len(output)
        for question in output:
            question.click()
        sleep(1.5)
        output_before_visualize = driver.find_elements_by_xpath('//div[@class="match-mod-horizontal-padding hide-focus-ring cbphWd"]')
        for question in output_before_visualize[len(output):-4]:        #visualize the questions that are not showed on the page
            driver.execute_script("arguments[0].scrollIntoView();", question)
            sleep(sleep_time)
        output_after_visualize = driver.find_elements_by_xpath('//div[@class="match-mod-horizontal-padding hide-focus-ring cbphWd"]')
        return(output_after_visualize)

def get_the_question_list(question,deepth,sleep_time):
    output_3 = single_question_scraper(question,deepth,sleep_time)
    for question in output_3:
        print(question.text)

if __name__ =="__main__":
    options = webdriver.FirefoxOptions()
    options.add_argument('-headless')
    driver = webdriver.Firefox(executable_path="/Users/huanranyixin/Desktop/geckodriver")
    #driver = webdriver.Firefox(executable_path="/Users/huanranyixin/Desktop/geckodriver",firefox_options=options)
    driver.get('https://www.google.com')
    # question = 'apple stock ?'
    #get_the_question_list('facebook stock ?',2,0.15)
    l = ['apple stock ?','facebook stock ?']
    scraper(l,2,0.15)
    driver.quit()

    