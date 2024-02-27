import os
import requests
from bs4 import BeautifulSoup

def download_all_images(url, folder='downloaded_images'):
    # Make a request to fetch the content of the page
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all image tags by class
    img_tags = soup.find_all('img')


    # Create a folder to store the images if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Loop over each image tag and download the image
    for img_tag in img_tags:
        # BLA_wBUJrga_SkfJ8won has to be in classes
        
        img_url = img_tag['src']
        # Sometimes, the image URL can be relative, so we have to make it absolute
        if not img_url.startswith(('data:image', 'http', 'https')):
            img_url = os.path.join(url, img_url)
        filename = folder + '/' + os.path.basename(img_url).split('?')[0]
        with open(filename, 'wb') as f:
            img_data = requests.get(img_url).content
            f.write(img_data)

if __name__ == '__main__':
    # url = input('Enter the URL of the webpage: ')
    download_all_images("https://www.gettyimages.com/photos/elon-musk", 'training_data/elon_musk')
    print('Download complete.')
