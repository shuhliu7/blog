# Google scraper

描述：输入两个关键词+问号，运行程序，自动从google上摘取people also asks的问句，因此可以快速帮助公司获取大量问句。

### code and decsription
```
搜索单个问题
question 是要问的问题 example：'apple stock ?'
deepth 是提取问题数量层数 通常为两层 0层可提取4个问题
1层可提取15个问题 2层可提取50个问题
sleep_time是为了让页面有足够时间加载 默认参数0.15 如果程序没法跑 可以调高参数

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
```


```
提取多个问题
把需要提取的问题放入question_list
如果这个问题在google中没有搜索到相关问题，程序会print出something
wrong with 这个问句

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

```

```
运行部分
driver可以用#后的那一行，可以用headless隐藏打开浏览器的步骤，减少
运行时间，输出为在scraper方法中建立的text_list。所有被爬下来的问句
都在list里

if __name__ =="__main__":
    options = webdriver.FirefoxOptions()
    options.add_argument('-headless')
    driver = webdriver.Firefox(executable_path="/Users/huanranyixin/Desktop/geckodriver")
    #driver = webdriver.Firefox(executable_path="/Users/huanranyixin/Desktop/geckodriver",firefox_options=options)
    driver.get('https://www.google.com')
    l = ['apple stock ?','facebook stock ?']
    scraper(l,2,0.15)
    driver.quit()
```

![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)