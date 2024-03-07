from bs4 import BeautifulSoup
import urllib3
import requests
import os.path as p
import numpy as np
import datamine as de

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# from os.path import basename


def main():
    '''
    try:
        baseurl = 'https://www.oppai-av.com/'
        exturl = 'works/detail/'
        http = urllib3.PoolManager()
        # urls = ['46d31101b3aaf13d', 'a033fd8c98aa7715', 'bda208ffe58542e8',
        #         'ff698b4048394cd3', '59ba2d9afd113ab8', 'bb7aeb1dd4493181',
        #         'efdb4b8d6bc38689', '29fa4b55234743a2', '60b3c8e04dcd209f',
        #         '8c10cf4e92916f79', 'b7c327e3cfa76ef1', '9a54815450bce59b',
        #         'd0c14e404eba3101', '1404dc31162ff861', 'fdd1beddb470f401',
        #         '392694e906be8f7b', '0cdf0cf5821b4247', 'f6b302d829628ae8',
        #         'a0c4fcdb0dc5da49', '1138a4f155ebb9ea', '7640a74ff5f3f780',
        #         '76d255faebf25341', 'b99e9984a4720579', '4cfa482f7b94ca20',
        #         '05b51a30824b5abd', '077c9de926927504', 'bf7896d83a58a1eb',
        #         '3fa527eb1ce41333']
        urls = ['pppd940', 'pppd251', 'pppd245', 'pppd395', 'pppd315']
        for url in urls:
            url = baseurl + exturl + url
            response = http.request('GET', url)
            soup = BeautifulSoup(response.data, "lxml")
            tags = soup.find_all('img')
            for tag in tags:
                if ('-s') in tag['src'] or ('-pl') in tag['src']:
                    tag['src'] = tag['src'].replace('-s', '')
                    path = 'data/sex/' + tag['src'].split('/')[4]
                    if not p.exists(path):
                        with open(path, 'wb') as f:
                            f.write(requests.get(baseurl + tag['src']).content)
                            f.close()
                    else:
                        print('File already exist!')
        pass
    except Exception as e:
        print('Failed in main loop')
        print(str(e))
    '''
    x = np.linspace(0, 10, 11)
    print(x)


if __name__ == "__main__":
    main()
    pass
# --------------------------------------------------------
'''
import numpy as np
delta = 1e-5
trans_cov = delta/(1-delta)*np.eye(2)
print(trans_cov)
'''
# --------------------------------------------------------
# import qlib
# print(help(qlib))
# # PACKAGE CONTENTS
# '''
#     config
#     contrib (package)
#     data (package)
#     log
#     model (package)
#     portfolio (package)
#     tests (package)
#     utils (package)
#     workflow (package)
# '''
# # FUNCTIONS
# '''
#     init
#     init_from_yaml_conf: ??? what is yaml
# '''
# --------------------------------------------------------
