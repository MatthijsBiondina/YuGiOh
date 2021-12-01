from src.utils.tools import pyout

ou = ''
val = 0.

missed = 0

early = 0
with open('res/marktplaats_msg.txt', 'r') as f:
    for line in sorted(list(f)):
        price = float(line.split(" | ")[-1].replace("\n", ""))
        if price < 0:
            price = 0.
            missed += 1


        val += price

        set = line.split(" | ")[1]
        if "Legend of Blue Eyes White Dragon" in set:
            early += 1
        if "Metal Raiders" in set:
            early += 1
        if "Spell Ruler" in set:
            early += 1
        if "Pharaoh's Servant" in set:
            early += 1
        if "Starter Deck: Kaiba" in set:
            early += 1
        if "Starter Deck: Yugi" in set:
            early += 1

        if "Dark Crisis" in set:
            early +=1
        if "Magician's Force" in set:
            early += 1
        if "Pharaonic Guardian" in set:
            early += 1
        if "Legacy of Darkness" in set:
            early += 1
        if "Labyrinth of Nightmare" in set:
            early += 1
        if "Starter Deck: Pegasus" in set:
            early += 1
        if "Starter Deck: Joey" in set:
            early += 1

        ou += ' | '.join(line.split(" | ")[:-1]) + "\n"

# with open('res/marktplaats_v1.txt', 'w+') as f:
#     f.write(ou)

pyout()