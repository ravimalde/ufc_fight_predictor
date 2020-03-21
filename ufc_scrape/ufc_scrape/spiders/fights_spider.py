# -*- coding: utf-8 -*-
import scrapy


class FightsSpider(scrapy.Spider):
    name = 'fights'
    allowed_domains = ['www.ufcstats.com']
    start_urls = ['http://www.ufcstats.com/statistics/events/completed?page=all']

    def parse(self, response):
        events = response.xpath("//tr/td/i/a")
        for event in events:
            link = event.xpath(".//@href").get()
        
            yield response.follow(url = link, callback=self.parse_event)

    def parse_event(self, response):
        event_name = response.xpath("//h2/span/text()").extract()
        event_date = response.xpath("//div[@class='b-list__info-box b-list__info-box_style_large-width']/ul/li[1]/text()").extract()
        rows = response.xpath("(//table[@class='b-fight-details__table b-fight-details__table_style_margin-top b-fight-details__table_type_event-details js-fight-table'])/tbody/tr")
        for row in rows:
            f1_name = row.xpath(".//td[2]/p[1]/a/text()").get()
            f2_name = row.xpath(".//td[2]/p[2]/a/text()").get()
            f1_str = int(row.xpath(".//td[3]/p[1]/text()").get())
            f2_str = int(row.xpath(".//td[3]/p[2]/text()").get())
            f1_tds = int(row.xpath(".//td[4]/p[1]/text()").get())
            f2_tds = int(row.xpath(".//td[4]/p[2]/text()").get())
            f1_subs = int(row.xpath(".//td[5]/p[1]/text()").get())
            f2_subs = int(row.xpath(".//td[5]/p[2]/text()").get())
            f1_pass = int(row.xpath(".//td[6]/p[1]/text()").get())
            f2_pass = int(row.xpath(".//td[6]/p[2]/text()").get())
            weight_class = row.xpath(".//td[7]/p/text()").get()
            method = row.xpath(".//td[8]/p[1]/text()").get()
            finisher = row.xpath(".//td[8]/p[2]/text()").get()
            round_num = row.xpath(".//td[9]/p/text()").get()
            time = row.xpath(".//td[10]/p/text()").get()
            yield{
                'f1_name': f1_name,
                'f2_name': f2_name,
                'f1_str': f1_str,
                'f2_str': f2_str,
                'f1_tds': f1_tds,
                'f2_tds': f2_tds,
                'f1_subs': f1_subs,
                'f2_subs': f2_subs,
                'f1_pass': f1_pass,
                'f2_pass': f2_pass,
                'weight_class': weight_class,
                'method': method,
                'finisher': finisher,
                'round_num': round_num,
                'time': time,
                'event_name': event_name,
                'event_date': event_date
            }