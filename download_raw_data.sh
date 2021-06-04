wget -P raw_data https://gbfs.citibikenyc.com/gbfs/en/station_information.json
cat raw_data_urls.txt | xargs -n 1 -P 6 wget -P raw_data/
unzip 'raw_data/*.zip' -d raw_data/
mkdir process_data