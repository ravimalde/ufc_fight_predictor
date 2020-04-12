# -*- coding: utf-8 -*-
import scrapy


class FightersSpider(scrapy.Spider):
    name = 'fighters2'
    allowed_domains = ['www.ufcstats.com']
    start_urls = ['http://www.ufcstats.com/statistics/fighters']

    def parse(self, response):
        fighters = response.xpath("(//td[1][@class='b-statistics__table-col'])/a")
        for fighter in fighters:
            link = fighter.xpath(".//@href").get()
            yield response.follow(url = link, callback=self.parse_fighter)

    def parse_fighter(self, response):
        body_stats = "//ul[@class='b-list__box-list']"
        name = response.xpath("//h2/span[1]/text()").get()
        record = response.xpath("//h2/span[2]/text()").get()
        height = response.xpath(body_stats+"/li[1]/text()").extract()
        weight = response.xpath(body_stats+"/li[2]/text()").extract()
        reach = response.xpath(body_stats+"/li[3]/text()").extract()
        stance = response.xpath(body_stats+"/li[4]/text()").extract()
        dob = response.xpath(body_stats+"/li[5]/text()").extract()

        fight_stats_1 = "(//ul[@class='b-list__box-list b-list__box-list_margin-top'])[1]"
        slmp = response.xpath(fight_stats_1+"/li[1]/text()").extract()
        str_acc = response.xpath(fight_stats_1+"/li[2]/text()").extract()
        ssapm = response.xpath(fight_stats_1+"/li[3]/text()").extract()
        str_def = response.xpath(fight_stats_1+"/li[4]/text()").extract()

        fight_stats_2 = "(//ul[@class='b-list__box-list b-list__box-list_margin-top'])[2]"
        td_avg = response.xpath(fight_stats_2+"/li[2]/text()").extract()
        td_acc = response.xpath(fight_stats_2+"/li[3]/text()").extract()
        td_def = response.xpath(fight_stats_2+"/li[4]/text()").extract()
        sub_avg = response.xpath(fight_stats_2+"/li[5]/text()").extract()


        yield{
            'name': name,
            'record': record,
            'height': height,
            'weight': weight,
            'reach': reach,
            'stance': stance,
            'dob': dob,
            'slmp': slmp,
            'str_acc': str_acc,
            'ssapm': ssapm,
            'str_def': str_def,
            'td_avg': td_avg,
            'td_acc': td_acc,
            'td_def': td_def,
            'sub_avg': sub_avg
        }
