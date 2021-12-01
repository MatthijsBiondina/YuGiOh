import json
import os
from urllib.request import Request, urlopen, urlretrieve

from tqdm import tqdm

from src.utils.render import show_card
from src.utils.tools import pyout


def download_card_db(download=True):
    os.makedirs('res/card_database', exist_ok=True)

    req = Request("https://db.ygoprodeck.com/api/v7/cardinfo.php",
                  headers={'User-Agent': 'Mozilla/5.0'})

    bdata = urlopen(req).read()
    data = json.loads(bdata.decode('utf8'))

    with open('res/card_database/ygoprodeck_db.json', 'w+') as f:
        json.dump(data, f, indent=2)


def download_card_images():
    os.makedirs('res/card_database/images', exist_ok=True)

    with open('res/card_database/ygoprodeck_db.json', 'r') as f:
        data = json.load(f)['data']

    for card in tqdm(data):
        for ii, card_img in enumerate(card['card_images']):
            fname = f"res/card_database/images/{card['id']}_{str(ii).zfill(2)}.jpg"

            if os.path.isfile(fname):
                continue
            else:
                try:
                    urlretrieve(card['card_images'][ii]['image_url'],
                                fname)
                except Exception:
                    pyout(fname)


def bin_cards():
    with open('res/card_database/ygoprodeck_db.json', 'r') as f:
        data = json.load(f)['data']

    types = {}
    for card in data:
        types[card['type']] = None
    types = list(types.keys())

    bins = {t: [c for c in data if c['type'] == t] for t in types}

    def merge(tgt, src):
        bins[tgt].extend(bins[src])
        bins.pop(src)
        types.remove(src)

    merge('Effect Monster', 'Flip Effect Monster')
    merge('Effect Monster', 'Union Effect Monster')
    merge('Effect Monster', 'Tuner Monster')
    merge('Synchro Monster', 'Synchro Tuner Monster')
    merge('Effect Monster', 'Gemini Monster')
    merge('Normal Monster', 'Normal Tuner Monster')
    merge('Effect Monster', 'Spirit Monster')
    merge('Ritual Monster', 'Ritual Effect Monster')

    for type_ in types:
        for card in data:
            if card['type'] == type_:
                pyout(type_)
                show_card(card['id'])
                break

    pyout()


# with open("res/card_database/ygoprodeck_db.json", 'r') as f:
#     data_old = json.load(f)
#
# with open("res/card_database/ygoprodeck_db_old.json", 'w+') as f:
#     json.dump(data_old, f)
#
# data_old = data_old['data']
#
# data_new = []
# for card in data_old:
#     if card['type'] not in ('Skill Card', 'Token'):
#         data_new.append(card)
#
# data_new = {'data':data_new}
#
# with open("res/card_database/ygoprodeck_db.json", 'w') as f:
#     json.dump(data_new, f)