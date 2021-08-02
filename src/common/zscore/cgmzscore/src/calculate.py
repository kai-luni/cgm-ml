"""https://www.who.int/childgrowth/standards/Chap_7.pdf"""
from decimal import Decimal as D


class Zscore:
    """Class to do calculation of z score"""

    def __init__(self, skew: int, median: int, coff: int, measurement: int):
        self.skew = skew
        self.median = median
        self.coff = coff
        self.measurement = measurement

    def calc_stdev(self, number: int):
        """This is not usual Standard Deviation please visit above PDF attached for clarification"""
        value = (1 + (self.skew * self.coff * number))**(1 / self.skew)
        stdev = self.median * value
        return stdev

    def z_score_measurement(self) -> float:
        """
         Z score
                  [y/M(t)]^L(t) - 1
           Zind =  -----------------
                      S(t)L(t)

                |       Zind            if |Zind| <= 3
                |
                |
                |       y - SD3pos
        Zind* = | 3 + ( ----------- )   if Zind > 3
                |         SD23pos
                |
                |
                |
                |        y - SD3neg
                | -3 + ( ----------- )  if Zind < -3
                |          SD23neg
        """

        numerator = (self.measurement / self.median)**self.skew - D(1.0)
        denominator = self.skew * self.coff
        z_score = numerator / denominator

        if D(z_score) > D(3):
            SD2pos = self.calc_stdev(2)
            SD3pos = self.calc_stdev(3)

            SD23pos = SD3pos - SD2pos

            z_score = 3 + ((self.measurement - SD3pos) / SD23pos)

            z_score = float(z_score.quantize(D('0.01')))

        elif D(z_score) < -3:
            SD2neg = self.calc_stdev(-2)
            SD3neg = self.calc_stdev(-3)

            SD23neg = SD2neg - SD3neg

            z_score = -3 + ((self.measurement - SD3neg) / SD23neg)
            z_score = float(z_score.quantize(D('0.01')))

        else:
            z_score = float(z_score.quantize(D('0.01')))

        return z_score
