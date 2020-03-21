# -*- coding: utf-8 -*-
import scrapy


class FightersSpider(scrapy.Spider):
    name = 'fighters'
    allowed_domains = ['www.ufcstats.com']
    start_urls = ['http://ufcstats.com/statistics/fighters/']

    def parse(self, response):
        fighters = response.xpath("(//td[1][@class='b-statistics__table-col'])/a")
        for fighter in fighters:
            link = fighter.xpath(".//@href").get()
            yield response.follow(url = link, callback=self.parse_fighter)

    def parse_fighter(self, response):
        height = response.xpath("(//ul[@class='b-list__box-list'])/li[1]")
        yield{
            'height': height
        }
