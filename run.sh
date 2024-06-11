#python main.py --nt 10000 --nb 100 -ne 1000 --noclip
#python main.py -nt 5000 -nb 100 -ne 400 -lr 0.01 -pa 5 -af -ni
#python main.py -nt 4500 -nb 100 -ne 400 -lr 0.01 -pa 5
python main.py -nt 4999 -nb 100 -ne 400 -pa 10 -td -ai -at
python main.py -nt 4910 -nb 100 -ne 400  -td --add-inventory --add-time --plot-after 10 --book-size 1000
python main.py -nt 4950 -nb 100 -ne 400 --plot-after 10 -td --add-inventory --add-time --book-size 5000