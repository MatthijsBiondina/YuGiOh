import re
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup as BS

from src.utils.tools import pyout

base_url = "https://www.cardmarket.com/en/YuGiOh/Products/Singles"


def mod_card_name(card_name):
    if "Shien-s" in card_name:
        card_name = card_name.replace('Shien-s', 'Shiens')
    if "Amazoness-Archer" in card_name:
        card_name = card_name.replace('Amazoness-Archer', 'Amazon-Archer')
    if "Hidden-Spellbook" in card_name:
        card_name = card_name.replace('Hidden-Spellbook', 'Hidden-Book-of-Spell')
    if "Spellbook-Organization":
        card_name = card_name.replace("Spellbook-Organization", "Pigeonholing-Books-of-Spell")
    if "Man-Thro-Tro-":
        card_name = card_name.replace('Man-Thro-Tro-', "ManThro-Tro")
    return card_name


def mod_set_name(set_name):
    h = set_name.split('-')

    if set_name == 'Legendary-Collection-3-Yugis-World-Mega-Pack':
        return "Legendary-Collection-3-Mega-Pack"
    elif set_name == 'Legendary-Collection-4-Joeys-World-Mega-Pack':
        return "Legendary-Collection-4-Mega-Pack"
    elif set_name == 'Saga-of-Blue-Eyes-White-Dragon-Structure-Deck':
        return 'Structure-Deck-Saga-of-BlueEyes-White-Dragon'
    elif len(h) > 2 and h[-2] == "Structure" and h[-1] == "Deck":
        return f"Structure-Deck-{'-'.join(h[:-2])}"
    else:
        return set_name


def scrape_card_mint_price(name, release):
    set_name = release['set_name']
    while '(' in set_name:
        start = set_name.find("(")
        end = set_name.find(")")
        set_name = set_name[:start] + set_name[end:]
    set_name = re.sub('[ ]', '-', release['set_name']).replace("-Volume-", '-')
    set_name = re.sub('[^a-zA-Z0-9-]', '', set_name)
    while '--' in set_name:
        set_name = set_name.replace('--', '-')
    set_name = mod_set_name(set_name)

    card_name = re.sub('[ \'!]', '-', name).replace("-Volume-", '-')
    card_name = re.sub('[^a-zA-Z0-9-]', '', card_name)
    while '--' in card_name:
        card_name = card_name.replace('--', '-')

    # card_name = re.sub('[^a-zA-Z0-9 -]', '', name).replace(' ', '-')
    card_name = mod_card_name(card_name)

    try:
        req = Request(f"{base_url}/{set_name}/{card_name}",
                      headers={'User-Agent': 'Mozilla/5.0'})
        data = urlopen(req).read()
        soup = BS(data, 'html.parser')
        soup = soup.find(id='tabContent-info')
        dl = soup.find_all('dl')[0]
        dt = [d.contents[0] for d in dl.find_all('dt')]
        dd = [d.contents[0] for d in dl.find_all('dd')]

        data = {dt_: dd_ for dt_, dd_ in zip(dt, dd)}

        price = data['30-days average price'].string
        price = float(re.sub('[^0-9.]', '', price.replace(',', '.')))

        return price
    except AttributeError:
        try:
            req2 = Request(f"{base_url}/{set_name}/{card_name}-V-1",
                           headers={'User-Agent': 'Mozilla/5.0'})
            data = urlopen(req2).read()
            soup = BS(data, 'html.parser')
            soup = soup.find(id='tabContent-info')

            dl = soup.find_all('dl')[0]
            dt = [d.contents[0] for d in dl.find_all('dt')]
            dd = [d.contents[0] for d in dl.find_all('dd')]

            data = {dt_: dd_ for dt_, dd_ in zip(dt, dd)}

            price = data['30-days average price'].string
            price = float(re.sub('[^0-9.]', '', price.replace(',', '.')))

            return price
        except AttributeError:
            pyout(req.full_url)
            return -1.
