# -*- coding: utf-8 -*-
import scrapy
import string


class FightersSpider(scrapy.Spider):

    name = 'fighters'
    allowed_domains= ['ufcstats.com']

    def start_requests(self):
        for letter in string.ascii_lowercase:
            yield scrapy.Request(url='http://www.ufcstats.com/statistics/fighters?char=' + letter + '&page=all', callback=self.parse)
        

    def parse(self, response):
        fighter_links = response.xpath("//td[@class ='b-statistics__table-col']//@href").extract()
        for link in fighter_links:
            yield scrapy.Request(url=link, callback=self.parse_fighter)

    def parse_fighter(self, response):
        body_stats = "(//ul[@class='b-list__box-list'])[1]"
        name = response.xpath("//h2/span[1]/text()").get()
        record = response.xpath("//h2/span[2]/text()").get()
        height = response.xpath(body_stats+"/li[1]/text()[2]").getall()
        weight = response.xpath(body_stats+"/li[2]/text()[2]").getall()
        reach = response.xpath(body_stats+"/li[3]/text()[2]").getall()
        stance = response.xpath(body_stats+"/li[4]/text()[2]").getall()
        dob = response.xpath(body_stats+"/li[5]/text()[2]").getall()

        fight_links = response.xpath("//tr//@data-link").extract()

        for link in fight_links:
            yield scrapy.Request(url=link, callback=self.parse_fights, dont_filter=True, meta={'name': name,
                                                                                                'record': record,
                                                                                                'height': height,
                                                                                                'weight': weight,
                                                                                                'reach': reach,
                                                                                                'stance': stance,
                                                                                                'dob': dob,})

    def parse_fights(self, response):

        name = response.meta['name']
        record = response.meta['record']
        height = response.meta['height']
        weight = response.meta['weight']
        reach = response.meta['reach']
        stance = response.meta['stance']
        dob = response.meta['dob']

        event_name = response.xpath("//h2/a/text()").get()   
        f1_name_a = response.xpath("(//h3[@class='b-fight-details__person-name'])[1]/a/text()").get()
        f1_name_span = response.xpath("(//h3[@class='b-fight-details__person-name'])[1]/span/text()").get()
        f2_name_a = response.xpath("(//h3[@class='b-fight-details__person-name'])[2]/a/text()").get()
        f2_name_span = response.xpath("(//h3[@class='b-fight-details__person-name'])[2]/span/text()").get()
        weightclass = response.xpath("//i[@class='b-fight-details__fight-title']/text()").get()
        rounds = response.xpath("(//i[@class='b-fight-details__text-item'])[1]/text()").extract()
        time = response.xpath("(//i[@class='b-fight-details__text-item'])[2]/text()").extract()
        f1_result = response.xpath("(//div[@class='b-fight-details__person'])[1]/i/text()").get()
        f2_result = response.xpath("(//div[@class='b-fight-details__person'])[2]/i/text()").get()
        f1_kd = response.xpath("//tbody[1]/tr[1]/td[2]/p[1]/text()").get()
        f2_kd = response.xpath("//tbody[1]/tr[1]/td[2]/p[2]/text()").get()
        f1_sig_str = response.xpath("//tbody[1]/tr[1]/td[3]/p[1]/text()").get()
        f2_sig_str = response.xpath("//tbody[1]/tr[1]/td[3]/p[2]/text()").get()
        f1_sig_str_perc = response.xpath("//tbody[1]/tr[1]/td[4]/p[1]/text()").get()
        f2_sig_str_perc = response.xpath("//tbody[1]/tr[1]/td[4]/p[2]/text()").get()
        f1_tot_str = response.xpath("//tbody[1]/tr[1]/td[5]/p[1]/text()").get()
        f2_tot_str = response.xpath("//tbody[1]/tr[1]/td[5]/p[2]/text()").get()
        f1_td = response.xpath("//tbody[1]/tr[1]/td[6]/p[1]/text()").get()
        f2_td = response.xpath("//tbody[1]/tr[1]/td[6]/p[2]/text()").get()
        f1_td_perc = response.xpath("//tbody[1]/tr[1]/td[7]/p[1]/text()").get()
        f2_td_perc = response.xpath("//tbody[1]/tr[1]/td[7]/p[2]/text()").get()
        f1_sub_att = response.xpath("//tbody[1]/tr[1]/td[8]/p[1]/text()").get()
        f2_sub_att = response.xpath("//tbody[1]/tr[1]/td[8]/p[2]/text()").get()
        f1_pass = response.xpath("//tbody[1]/tr[1]/td[9]/p[1]/text()").get()
        f2_pass = response.xpath("//tbody[1]/tr[1]/td[9]/p[2]/text()").get()
        f1_rev = response.xpath("//tbody[1]/tr[1]/td[10]/p[1]/text()").get()
        f2_rev = response.xpath("//tbody[1]/tr[1]/td[10]/p[2]/text()").get()
        f1_sig_str_head = response.xpath("(//tbody[@class='b-fight-details__table-body'])[3]/tr/td[4]/p[1]/text()").get()
        f2_sig_str_head = response.xpath("(//tbody[@class='b-fight-details__table-body'])[3]/tr/td[4]/p[2]/text()").get()
        f1_sig_str_body = response.xpath("(//tbody[@class='b-fight-details__table-body'])[3]/tr/td[5]/p[1]/text()").get()
        f2_sig_str_body = response.xpath("(//tbody[@class='b-fight-details__table-body'])[3]/tr/td[5]/p[2]/text()").get()
        f1_sig_str_leg = response.xpath("(//tbody[@class='b-fight-details__table-body'])[3]/tr/td[6]/p[1]/text()").get()
        f2_sig_str_leg = response.xpath("(//tbody[@class='b-fight-details__table-body'])[3]/tr/td[6]/p[2]/text()").get()
        f1_sig_str_dist = response.xpath("(//tbody[@class='b-fight-details__table-body'])[3]/tr/td[7]/p[1]/text()").get()
        f2_sig_str_dist = response.xpath("(//tbody[@class='b-fight-details__table-body'])[3]/tr/td[7]/p[2]/text()").get()
        f1_sig_str_clinch = response.xpath("(//tbody[@class='b-fight-details__table-body'])[3]/tr/td[8]/p[1]/text()").get()
        f2_sig_str_clinch = response.xpath("(//tbody[@class='b-fight-details__table-body'])[3]/tr/td[8]/p[2]/text()").get()
        f1_sig_str_ground = response.xpath("(//tbody[@class='b-fight-details__table-body'])[3]/tr/td[9]/p[1]/text()").get()
        f2_sig_str_ground = response.xpath("(//tbody[@class='b-fight-details__table-body'])[3]/tr/td[9]/p[2]/text()").get()

        yield{
            'name': name,
            'record': record,
            'height': height,
            'weight': weight,
            'reach': reach,
            'stance': stance,
            'dob': dob,
            'event_name': event_name,
            'f1_name_a': f1_name_a,
            'f1_name_span': f1_name_span,
            'f2_name_a': f2_name_a,
            'f2_name_span': f2_name_span,
            'weightclass': weightclass,
            'rounds': rounds,
            'time': time,
            'f1_result': f1_result,
            'f2_result': f2_result,
            'f1_kd': f1_kd,
            'f2_kd': f2_kd,
            'f1_sig_str': f1_sig_str,
            'f2_sig_str': f2_sig_str,
            'f1_sig_str_perc': f1_sig_str_perc,
            'f2_sig_str_perc': f2_sig_str_perc,
            'f1_tot_str': f1_tot_str,
            'f2_tot_str': f2_tot_str,
            'f1_td': f1_td,
            'f2_td': f2_td,
            'f1_td_perc': f1_td_perc,
            'f2_td_perc': f2_td_perc,
            'f1_sub_att': f1_sub_att,
            'f2_sub_att': f2_sub_att,
            'f1_pass': f1_pass,
            'f2_pass': f2_pass,
            'f1_rev': f1_rev,
            'f2_rev': f2_rev,
            'f1_sig_str_head': f1_sig_str_head,
            'f2_sig_str_head': f2_sig_str_head,
            'f1_sig_str_body': f1_sig_str_body,
            'f2_sig_str_body': f2_sig_str_body,
            'f1_sig_str_leg': f1_sig_str_leg,
            'f2_sig_str_leg': f2_sig_str_leg,
            'f1_sig_str_dist': f1_sig_str_dist,
            'f2_sig_str_dist': f2_sig_str_dist,
            'f1_sig_str_clinch': f1_sig_str_clinch,
            'f2_sig_str_clinch': f2_sig_str_clinch,
            'f1_sig_str_ground': f1_sig_str_ground,
            'f2_sig_str_ground': f2_sig_str_ground
        }

