import csv
import json
import random
import time

from tqdm import tqdm

from src.scraping.cardmarket import scrape_card_mint_price
from src.utils.tools import pyout

mint_prices = {}
errors = 1
while errors > 0:
    errors = 0
    with open('res/card_database/ygoprodeck_db.json', 'r') as f:
        db = json.load(f)['data']
        db = {entry['id']: entry for entry in db}

    with open('res/store', 'r') as f:
        reader = csv.reader(f)
        store = [row for row in reader]

    msg = ""

    modifier = {'Near Mint': 1., 'Excellent': 0.9, 'Good': 0.8, 'Light Played': 0.7, 'Played': 0.6,
                'Poor': 0.}

    store.sort(key=lambda x: int(x[0]))


    for card in tqdm(store):
        ii = int(card[0])
        name = db[ii]['name']
        release = [s for s in db[ii]['card_sets'] if s['set_code'] == card[1]][0]
        cset = release['set_name']
        cond = card[3]

        try:
            try:
                mint = mint_prices[ii]
            except KeyError:
                mint_prices[ii] = scrape_card_mint_price(name, release)
                mint = mint_prices[ii]
                time.sleep(random.uniform(3, 7))

            valu = f"{float(mint) * modifier[cond]:.2f}"

            msg += f"{name} | {cset} | {cond} | {valu}\n"
        except Exception as e:
            # pyout(e)
            errors += 1
            time.sleep(15)

    with open('res/marktplaats_msg.txt', 'w+') as f:
        f.write(msg[:-1])

    pyout(f"Errors: {errors}")


