from  bs4 import BeautifulSoup
import requests
import pandas as pd
import re
import datetime
import dateparser


def build_urls():
    """
    Method to create a list of urls that need to be scraped
    """
    urls = []
    base = 'https://swappa.com/mobile/buy/'
    models = ['apple-iphone-11','apple-iphone-x','apple-iphone-8','apple-iphone-7-plus','apple-iphone-7','apple-iphone-xs-max',
    'apple-iphone-xs','apple-iphone-xr','apple-iphone-8-plus','apple-iphone-6s-plus','apple-iphone-6s']
    carriers = ['att','verizon','sprint','unlocked','t-mobile']

    for car in carriers:
        for mod in models:
            urls.append(base + mod +'/'+ car)

    return urls

class ScrapeURL():
    """
    Class to scrape data from a url and returns a processed DataFrame.
    """
    def __init__(self, url):
        self.url = url
        self.df = pd.DataFrame()

    def process_df(self):
        """
        Processes the dataframe of recently sold phone
        """
        self.df['price'] = self.df['price'].apply(lambda x: int(x[1:]))
        self.df['storage'] = self.df['storage'].apply(lambda x: int(x.replace('GB','')))
        self.df['carrier'] = self.df['url'].apply(lambda x: x.split('/')[-1])
        self.df['model'] = self.df['url'].apply(lambda x: x.split('/')[-2])
        self.df['date'] = self.df['date'].apply(lambda x: dateparser.parse(x.replace('Sold on ','')).date().isoformat())

    def pull_data(self):
        """
        Function looks for the recently sold section in self.url and pulls price, condition, storage, color, date and listing data.
        """
        page = requests.get(self.url, headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36'})
        if page.status_code == 404:
            return pd.DataFrame(columns=['price', 'condition', 'storage', 'color', 'date', 'url'])

        soup = BeautifulSoup(page.text, features="html5lib")
        recently_sold = soup.findAll(class_='col-xs-12 col-sm-8 col-sm-push-4 col-md-9 col-md-push-3')
        all_links = soup.findAll('a' , href=True)

        price_data = [x.text for x in recently_sold[0].findAll(class_='price')][::2]
        condition_data = [x.text for x in recently_sold[0].findAll(class_='condition_label')]
        storage_data = [x.text for x in recently_sold[0].findAll(class_='storage_label')]
        color_data = [x.text for x in recently_sold[0].findAll(class_='color_label')]
        date_data = [x.text for x in recently_sold[0].findAll("span") if 'Sold' in x.text]
        listing_url = [x['href'] for x in all_links if '/listing/view/' in x['href']][-5:]

        self.df = pd.DataFrame({'price':price_data,'storage':storage_data,'condition':condition_data,'color':color_data,'date':date_data, 'listing_url':listing_url})
        self.df['url'] = self.url

    def run(self):
        self.pull_data()
        self.process_df()
        return self.df

def main():
    urls = build_urls()
    all_df = pd.DataFrame(columns=['price', 'condition', 'storage', 'color', 'date','listing_url', 'url'])

    for url in urls:
        df = ScrapeURL(url).run()
        all_df = pd.concat([all_df, df])

    all_df.to_csv("data_{}.csv".format(datetime.datetime.now().strftime("%m_%d_%Y")),index=False)


if __name__ == '__main__':
    while True:
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start, {}".format(current_time))
    main()
    print("Stop, {}".format(current_time))
    time.sleep(10800)
