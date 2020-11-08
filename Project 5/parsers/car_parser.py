import requests
from bs4 import BeautifulSoup
import os
import json
from tqdm import tqdm
from threading import Thread
import concurrent.futures


def get_data_async(start_year=1960, marks=None):
    if marks is None:
        raise Exception('Marks have to be set!')

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        for mark in marks:
            current_parser = CarsParser(start_year=start_year)
            futures.append(executor.submit(current_parser.get_model_data, model=mark))
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

   

class CarsParser:
    def __init__(self, site='auto.ru', start_year=1990, end_year=2020) -> None:
        self.start_year = start_year
        self.end_year = end_year
        self.site = site if site in ['auto.ru'] else 'auto.ru'
        self.marks_ = []  # марки автомобилей для парсера
        self.get_marks()  # заполним словарь марок для парсинга
        self.offers = []
        self.path = os.path.dirname(os.path.realpath(__file__))

        # Запишем заголовки для запроса на авто.ру
        self.headers = self.prepare_headers()

    # Подготовим заголовки
    def prepare_headers(self):
        headers = '''
Host: auto.ru
Connection: keep-alive
Content-Length: 154
Pragma: no-cache
Cache-Control: no-cache
x-requested-with: fetch
x-client-date: 1604755010487
x-csrf-token: f6f0f34be8cfdd7ff0bd22851d5988e2c57cd4c953e7dfa9
x-page-request-id: d31d80148963a69d1925d76efaf69ec4
content-type: application/json
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36 OPR/72.0.3815.186
x-client-app-version: 202011.06.173706
Accept: */*
Origin: https://auto.ru
Sec-Fetch-Site: same-origin
Sec-Fetch-Mode: same-origin
Sec-Fetch-Dest: empty
Accept-Encoding: gzip, deflate, br
Accept-Language: ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7
Cookie: _csrf_token=f6f0f34be8cfdd7ff0bd22851d5988e2c57cd4c953e7dfa9; autoru_sid=a%3Ag5fa674f0234roab1hp7a2sismombg2q.7e66a9df6689f452d36151c294a3e610%7C1604744432607.604800.KwL1iB2fmfrve4pwfXnLwQ.O7CUm0ZQlt4mmVf-LG9gUVDvKgnJbm2wfhqAiqu7sO0; autoruuid=g5fa674f0234roab1hp7a2sismombg2q.7e66a9df6689f452d36151c294a3e610; suid=6c07b707f6cd3f5edfc51d025ed1bc4b.8d92ebdab9df82ed833be19559590ea4; from=direct; yuidcs=1; crookie=AGRrOr99jaz+1RUrkelbWhbJ6GsitmKopD2cXuR8CzJVQESjUz6qjks84rEZ6fCRDVVKLqN7J9ZN+zjiKrfm7+20IhA=; cmtchd=MTYwNDc0NDQzNjUyNQ==; popup_new_user=new; proven_owner_popup=1; _ym_uid=16047444391066413935; _ym_isad=1; bltsr=1; yuidlt=1; yandexuid=3231444291604738211; counter_ga_all7=2; _ym_wasSynced=%7B%22time%22%3A1604744534764%2C%22params%22%3A%7B%22eu%22%3A0%7D%2C%22bkParams%22%3A%7B%7D%7D; gdpr=0; _ga=GA1.2.282058101.1604744536; _gid=GA1.2.264216814.1604744536; index-selector-tab=marks; X-Vertis-DC=myt; _ym_visorc_22753222=b; from_lifetime=1604754983938; _gat_fixprocent=1; _ym_d=1604754987
'''.strip().split('\n')

        dict_header = {}

        for header in headers:
            key, value = header.split(': ')
            dict_header[key] = value

        return dict_header

    # Получаем список марок для парсера
    def get_marks(self):
        path = os.path.dirname(os.path.realpath(__file__))
        with open(path+'\\'+self.site+'_marks.html', encoding='utf8') as reader:
            marks = reader.read()
        
        soup = BeautifulSoup(marks, 'html.parser')

        if self.site == 'auto.ru':
            for link in soup.find_all('a', class_='IndexMarks__item', href=True):
                link_url = link['href'].split('/')
                self.marks_.append(link_url[-3].upper())
        # Другие сайты для парсинга
        else:
            pass

    # Получаем данные с авто.ру
    def get_model_data(self, model):
    
        # Забираем все объявления по марке и годам
        for year in range(self.start_year, self.end_year + 1):

            print(f'Getting {year} for mark {model}')

            params = {"year_from":year,"year_to":year,"catalog_filter":[{"mark":model}],
                    "section":"all","category":"cars","output_type":"list","geo_radius":200,"geo_id":[213], "page": 1,
                    'sort': "fresh_relevance_1-desc", "top_days":"900"}

            try:
                total_page_count = 0
                request = requests.post('https://auto.ru/-/ajax/desktop/listing/', headers=self.headers, json=params).json()
                total_page_count = request['pagination']['total_page_count']
                
                if request['offers'] != []:
                    self.offers.append(request['offers'])           
           
                page = 2
                for i in range(page, total_page_count+1):
                    print(f'Getting page {i}/{total_page_count}')
                    params = {"year_from":year,"year_to":year,"catalog_filter":[{"mark":model}],
                    "section":"all","category":"cars","output_type":"list","geo_radius":200,"geo_id":[213], "page": i,
                    'sort': "fresh_relevance_1-desc", "top_days":"900"}

                    request = requests.post('https://auto.ru/-/ajax/desktop/listing/', headers=self.headers, json=params).json()
                    
                    if request['offers'] != []:
                        self.offers.append(request['offers'])  

            except Exception as e:
                print(e)     
        
        # Пишем полученные данные в JSON
        self.model_to_json(model)

        # Очистим память
        self.offers.clear()

        print(f'Parsing {model} done!')
   
    # Считаем количество данных в датасете модели
    def get_model_count(self, model):
        count = 0
        for m in model:
            count += len(m)

        return count

    # Сохраняем данные модели в json
    def model_to_json(self, model):
        with open(self.path+'\\data\\'+model+'.json', 'w', encoding='utf8') as file:
            json.dump(self.offers, file)

    # Загружаем модель в json
    def load_model(self, model):
        with open(self.path+'\\data\\'+model+'.json', encoding='utf8') as file:
            model_data = json.load(file)

        return model_data

    # Создаем датасет для модели
    def create_dataset(self, model, filename='dataset'):
        columns = ['bodyType', 'brand', 'color', 'fuelType', 'modelDate', 'name',
       'numberOfDoors', 'productionDate', 'vehicleConfiguration',
       'vehicleTransmission', 'engineDisplacement', 'enginePower',
       'description', 'mileage', 'Комплектация', 'Привод', 'Руль', 'Состояние',
       'Владельцы', 'ПТС', 'Таможня', 'Владение', 'price', 'currency']

        try:
            # Пишем в csv
            with open(self.path+'\\data\\'+filename+'.csv', 'a', encoding='utf8') as file:                
                file.write('|'.join(columns) + '\n') 

                for item in tqdm(model):
                    if len(item) == 0:
                        continue

                    for m in item:                        
                        row = []
                        bodyType = m['vehicle_info']['configuration']['human_name'].lower()
                        brand = m['vehicle_info']['mark_info']['name']
                        color = m['color_hex']
                        fuelType = m['vehicle_info']['tech_param']['engine_type']
                        modelDate = m['vehicle_info']['super_gen']['year_from']
                        name = m['vehicle_info']['tech_param']['human_name']
                        numberOfDoors = m['vehicle_info']['configuration']['doors_count']
                        productionDate = m['documents']['year']
                        vehicleConfiguration = m['vehicle_info']['configuration']['body_type'] + ' ' + m['vehicle_info']['tech_param']['transmission'] + ' ' + m['vehicle_info']['tech_param']['human_name'].split()[0]
                        vehicleTransmission = m['vehicle_info']['tech_param']['transmission']
                        engineDisplacement = m['vehicle_info']['tech_param']['human_name'].split()[0]
                        enginePower = m['vehicle_info']['tech_param']['power']
                        description = m['description'].replace('\n', ' ').replace('\r', '').replace('|', '') if 'description' in m else ''
                        mileage = m['state']['mileage']
                        complectation = m['vehicle_info']['equipment'] if 'equipment' in m['vehicle_info'] else ''
                        gear_type = m['vehicle_info']['tech_param']['gear_type']
                        steer = m['vehicle_info']['steering_wheel']
                        health = ''
                        owners = m['documents']['owners_number'] if 'owners_number' in m['documents'] else ''
                        pts = m['documents']['pts_original'] if 'pts_original' in m['documents'] else ''
                        custom_cleared = m['documents']['custom_cleared']
                        ownage = str(m['documents']['purchase_date']['year']) + ' ' + str(m['documents']['purchase_date']['month']) if 'purchase_date' in m['documents'] else ''
                        price = m['price_info']['price'] if 'price' in m['price_info'] else ''
                        currency = m['price_info']['currency'] if 'currency' in m['price_info'] else ''

                        row = [ str(bodyType), 
                                str(brand),
                                str(color),
                                str(fuelType),
                                str(modelDate),
                                str(name),
                                str(numberOfDoors),
                                str(productionDate),
                                str(vehicleConfiguration),
                                str(vehicleTransmission),
                                str(engineDisplacement),
                                str(enginePower),
                                str(description),
                                str(mileage),
                                str(complectation),
                                str(gear_type),
                                str(steer),
                                str(health),
                                str(owners),
                                str(pts),
                                str(custom_cleared),
                                str(ownage),
                                str(price),
                                str(currency)
                        ]

                        file.write('|'.join(row) + '\n')
        except Exception as e:
            print(e)

parser = CarsParser()
car_marks = parser.marks_

#get_data_async(marks=car_marks)

model = parser.load_model('BMW')
parser.create_dataset(model, 'BMW')


