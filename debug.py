#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import re
import urllib
import json
import socket
import urllib.request
import urllib.parse
import urllib.error
import argparse
import sys
# 设置超时
import time

timeout = 5
socket.setdefaulttimeout(timeout)


class Crawler:
    # 睡眠时长
    __time_sleep = 0.1
    __amount = 0
    __start_amount = 0
    __counter = 1
    __page_count = 1

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}

    # 获取图片url内容等
    # t 下载图片时间间隔
    def __init__(self, t=0.1):
        self.time_sleep = t

    # 开始获取
    def __get_images(self, word, image_nums, start_image):

        count = image_nums
        first = start_image

        if not os.path.exists("E:/Python35/flower_photos/" + word):
            os.mkdir("E:/Python35/flower_photos/" + word)

        # async = content
        # q = 美女  搜索关键字
        # first = 118 开始条数
        # count = 35 显示数量
        reg = r'src="(https://tse.+?)"'
#        url = 'https://cn.bing.com/images/search?q=%E4%B9%94%E6%AC%A3&qs=n&form=QBILPG&sp=-1&pq=%E4%B9%94%E6%AC%A3&sc=8-2'
        url = 'https://www.bing.com/images/async?&async=content&q=' + urllib.parse.quote(word) + '&first=' + str(first) + '&count=' + str(count) + '&FORM=HDRSC2'
        url = 'https://www.bing.com/images/search?q=' + urllib.parse.quote(word) + '&first=' + str(first) + '&count=' + str(count)+ '&form=QBIR' + '&sp=-1'

        try:
            req = urllib.request.Request(url=url, headers=self.headers)
            page = urllib.request.urlopen(req)
            rsp = page.read().decode('utf8')
            links = re.findall(reg, rsp)
#            print (links)

        except UnicodeDecodeError as e:
            print(e)
            print('-----UnicodeDecodeErrorurl:', url)
        except urllib.error.URLError as e:
            print(e)
            print("-----urlErrorurl:", url)
        except socket.timeout as e:
            print(e)
            print("-----socket timout:", url)
#        else:
#           self.__save_image(links, word)
        finally:
            page.close()

        return links

    def __save_image(self, links, word):
        if not os.path.exists("E:/f/flower_photos/" + word):
            os.mkdir("E:/f/flower_photos/" + word)
        for link in links:
            try:
                time.sleep(self.time_sleep)
                urllib.request.urlretrieve(link, 'E:/f/flower_photos/' + word + '/%s.jpg' % (self.__counter - 1))
            except urllib.error.HTTPError as urllib_err:
                print(urllib_err)
                continue
            except Exception as err:
                time.sleep(1)
                print(err)
                print("产生未知错误，放弃保存")
                continue
            else:
                print(word+" +1,已有" + str(self.__counter) + "张图片")
                self.__counter += 1
        return


    def start(self, word, spider_image_num, start_image):
        """
        爬虫入口
        :param word: 抓取的关键词
        :param spider_image_num: 需要抓取图片数量
        :param start_image:起始图片
        :return:
        """
        print("********************************************************")
        print("********************************************************")

        time.sleep(1)
        self.__start_amount = start_image
        self.__amount = spider_image_num
        # url一次最多获取150个结果 所以采取分页，这里取140一页
        page_nums = int(spider_image_num/140 + 1)
#        print("page_nums",page_nums)
        while True:
            if page_nums == self.__page_count:
                image_nums = self.__amount%140 + 1
            else:
                image_nums = 140 + 1
            links = self.__get_images(word, image_nums, self.__start_amount + 140*(self.__page_count - 1))
            self.__save_image(links, word)
            self.__page_count += 1
            if page_nums < self.__page_count:
                break

        print("********************************************************")
        print("图片爬取结束！")
        print('\n')

def main(argv):
    parser = argparse.ArgumentParser(description="Image Downloader")
    parser.add_argument("keywords", type=str,help='Keywords to search. ("in quotes")')
    parser.add_argument("--number", "-n", type=int, default=100,
                        help="number of images download for the keywords.")
    parser.add_argument("--first", "-f", type=int, default=50,
                        help="start number of images download for the keywords.")

    crawler = Crawler(0.05)
    crawler.start(argv, 50, 5)

if __name__ == '__main__':
    main(sys.argv[1:])


