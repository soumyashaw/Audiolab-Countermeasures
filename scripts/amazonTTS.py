########################### Amazon TTS ##############################

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select

data = pd.read_csv("/Users/soumyashaw/validated.tsv", sep="\t")


driver = webdriver.Chrome()

driver.get("https://eu-central-1.console.aws.amazon.com/polly/home/SynthesizeSpeech")

time.sleep(2)

# Move to Root User
root_user_button = driver.find_element("id", "root_account_signin")
root_user_button.click()

time.sleep(2)

# Enter the email
email_fill = driver.find_element("id", "resolving_input")
email_fill.send_keys('soumya.shaw@cispa.de')

# Find the "Continue" button and click on it
continue_button = driver.find_element("id", "next_button")
continue_button.click()

_ = input("Waiting for Manual Captcha Entry")

# Find the password input field and enter your password
password_field = driver.find_element("id", "password")
password_field.send_keys('Rebashaw@2000')

# Find the "Sign in" button and click on it
signin_button = driver.find_element("id", "signin_button")
signin_button.click()

time.sleep(2)

# Select the Language
print("Select the Option: English Australian -> Olivia, Female")
_ = input()

globalCounter = 0
textarea = driver.find_element(By.ID, "formFieldForInputText")
clear_button = driver.find_element(By.CLASS_NAME, 'clearTextButton')
download_button = driver.find_element(By.CLASS_NAME, 'downloadButton')

for i in range(77):
    text = data['sentence'][globalCounter]

    clear_button.click()
    textarea.send_keys(text)
    time.sleep(2)
    download_button.click()
    time.sleep(1)

    print(i, globalCounter)
    globalCounter += 1

print()
print("Select the Option: English British -> Amy, Female")
_ = input()

for i in range(77):
    text = data['sentence'][globalCounter]

    clear_button.click()
    textarea.send_keys(text)
    time.sleep(2)
    download_button.click()
    time.sleep(1)

    print(i, globalCounter)
    globalCounter += 1

print()
print("Select the Option: English South African -> Ayanda, Female")
_ = input()

for i in range(77):
    text = data['sentence'][globalCounter]

    clear_button.click()
    textarea.send_keys(text)
    time.sleep(2)
    download_button.click()
    time.sleep(1)

    print(i, globalCounter)
    globalCounter += 1

print()
print("Select the Option: English US -> Danielle, Female")
_ = input()

for i in range(77):
    text = data['sentence'][globalCounter]

    clear_button.click()
    textarea.send_keys(text)
    time.sleep(2)
    download_button.click()
    time.sleep(1)

    print(i, globalCounter)
    globalCounter += 1

print()
print("Select the Option: English US -> Joanna, Female")
_ = input()

for i in range(76):
    text = data['sentence'][globalCounter]

    clear_button.click()
    textarea.send_keys(text)
    time.sleep(2)
    download_button.click()
    time.sleep(1)

    print(i, globalCounter)
    globalCounter += 1

print()
print("Select the Option: English US -> Ruth, Female")
_ = input()

for i in range(77):
    text = data['sentence'][globalCounter]

    clear_button.click()
    textarea.send_keys(text)
    time.sleep(2)
    download_button.click()
    time.sleep(1)

    print(i, globalCounter)
    globalCounter += 1

print()
print("Select the Option: English US -> Matthew, Male")
_ = input()

for i in range(77):
    text = data['sentence'][globalCounter]

    clear_button.click()
    textarea.send_keys(text)
    time.sleep(2)
    download_button.click()
    time.sleep(1)

    print(i, globalCounter)
    globalCounter += 1

print()
print("Select the Option: English US -> Stephen, Male")
_ = input()

for i in range(77):
    text = data['sentence'][globalCounter]

    clear_button.click()
    textarea.send_keys(text)
    time.sleep(2)
    download_button.click()
    time.sleep(1)

    print(i, globalCounter)
    globalCounter += 1

print()
print("Select the Option: French -> Lea, Female")
_ = input()

for i in range(77):
    text = data['sentence'][globalCounter]

    clear_button.click()
    textarea.send_keys(text)
    time.sleep(2)
    download_button.click()
    time.sleep(1)

    print(i, globalCounter)
    globalCounter += 1

print()
print("Select the Option: German -> Vicki, Female")
_ = input()

for i in range(77):
    text = data['sentence'][globalCounter]

    clear_button.click()
    textarea.send_keys(text)
    time.sleep(2)
    download_button.click()
    time.sleep(1)

    print(i, globalCounter)
    globalCounter += 1

print()
print("Select the Option: Spanish Castilian -> Lucia, Female")
_ = input()

for i in range(77):
    text = data['sentence'][globalCounter]

    clear_button.click()
    textarea.send_keys(text)
    time.sleep(2)
    download_button.click()
    time.sleep(1)

    print(i, globalCounter)
    globalCounter += 1

print()
print("Select the Option: Spanish Mexican -> Mia, Female")
_ = input()

for i in range(77):
    text = data['sentence'][globalCounter]

    clear_button.click()
    textarea.send_keys(text)
    time.sleep(2)
    download_button.click()
    time.sleep(1)

    print(i, globalCounter)
    globalCounter += 1

print()
print("Select the Option: Spanish US -> Lupe, Female")
_ = input()

for i in range(77):
    text = data['sentence'][globalCounter]

    clear_button.click()
    textarea.send_keys(text)
    time.sleep(2)
    download_button.click()
    time.sleep(1)

    print(i, globalCounter)
    globalCounter += 1

# Find the text input field and enter the target text
#text_field = driver.find_element("id", "formFieldForInputText")

# Loop through the list of target texts and generate the corresponding audio files
#for itr in range(endFileNumber - startFileNumber):
#    x = df['sentence'][startFileNumber + itr]
#
#    audio = generate(
#        text=x,
#        voice=voices[random.randint(0, len(voices)-1)],
#        model="eleven_multilingual_v2"
#        )
#    
#    save(audio, r"C:\Users\Dell\Downloads\\" + str(df['path'][startFileNumber + itr]) )

    # Clear former Text
    #text_field.clear()

    #text_field.send_keys(x)

    # Find the "Download" button and click on it
    #download_button = driver.find_element(By.CLASS_NAME, 'downloadButton')
    #download_button.click()

    # Wait for the audio file to be generated and downloaded
    #time.sleep(5)

    #download_directory = r'C:\Users\Dell\Downloads'

    #latest_file = max(
        #[os.path.join(download_directory, f) for f in os.listdir(download_directory)],
        #key=os.path.getctime
    #)

    # Change the name of the downloaded file to the name of the target file name
    #os.rename(latest_file, r'C:\Users\Dell\Downloads\\' + str(df['path'][startFileNumber + itr]))

time.sleep(150)
time.sleep(50)

# Close the browser window
#driver.quit()



########################### Google TTS ##############################

