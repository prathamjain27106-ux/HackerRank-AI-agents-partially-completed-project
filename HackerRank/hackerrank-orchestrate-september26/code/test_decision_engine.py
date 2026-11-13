"""Focused synthetic tests for deterministic decision planning."""
from datetime import date, timedelta
from decimal import Decimal
import unittest
from data_loading import ClassifiedEvent, FinancialEvent, FinancialProfile, FinancialState, PaymentOption, Request, RequestContext
from forecast import ForecastEngine, RecurringCashFlow
from decision_engine import DecisionEngine
S=date(2026,1,1)
def ctx(balance="100", minimum="20", methods=("full_payment",), partial=True, events=(), options=(), maxmonths=None, protect=(), reduce=(), stop=()):
 p=FinancialProfile("u","USD",Decimal(balance),Decimal(minimum),(),protect,reduce,stop,methods,maxmonths); c=tuple(ClassifiedEvent(e, e.event_id.startswith("rec")) for e in events); st=FinancialState(c,(),(),(),(),c,(),(),(),())
 r=Request("r","u",S,"purchase",Decimal("80"),S+timedelta(days=30),partial,""); return RequestContext(r,p,st,tuple(options),(),())
def ev(i,amount,day,flex="fixed"):
 return FinancialEvent(i,"u","expense",i,"fun","debit",Decimal(amount),"USD",day,day,"scheduled",None,flex,Decimal("0") if flex!="fixed" else None)
def inc(i,amount,day): return FinancialEvent(i,"u","income",i,"salary","credit",Decimal(amount),"USD",day,day,"scheduled",None,"fixed",None)
def opt(i,amt,n,first,fee="0",total=None): return PaymentOption(i,"r","installments",Decimal(amt),n,first,30,Decimal(fee),Decimal(total or str(Decimal(amt)*n)))
class DecisionTests(unittest.TestCase):
 def test_full_now_partial_wait_and_earliest(self):
  d=DecisionEngine(ForecastEngine({})); self.assertEqual("affordable_now",d.decide(ctx()).affordability_status)
  c=ctx(balance="100",events=(ev("bill","50",S+timedelta(days=10)),inc("salary","80",S+timedelta(days=11))),methods=("partial_payment","full_payment")); x=d.decide(c); self.assertEqual(Decimal("30"),x.amount_safe_to_pay); self.assertEqual(S+timedelta(days=11),x.earliest_date_for_full_payment)
 def test_installment_preference_deadline_and_max_months(self):
  d=DecisionEngine(ForecastEngine({})); good=opt("a","45",2,S,total="90"); bad=opt("b","45",2,S,total="90")
  self.assertEqual("installments",d.decide(ctx(balance="110",methods=("installments",),options=(good,),maxmonths=2)).recommended_payment_method)
  self.assertEqual("not_recommended",d.decide(ctx(balance="110",methods=("installments",),options=(bad,),maxmonths=1)).recommended_payment_method)
 def test_preference_and_no_plan(self):
  d=DecisionEngine(ForecastEngine({})); o=opt("a","30",3,S,total="90")
  self.assertEqual("not_recommended",d.decide(ctx(balance="40",methods=("full_payment",),options=(o,),maxmonths=3)).recommended_payment_method)
 def test_changes_protected_and_max_three(self):
  d=DecisionEngine(ForecastEngine({})); es=tuple(ev(f"rec{i}","20",S-timedelta(days=1),"stoppable") for i in range(4)); flows=tuple(RecurringCashFlow(f"f{i}",S+timedelta(days=1),Decimal("20"),"USD","debit",90,1,f"rec{i}") for i in range(4))
  r=d.decide(ctx(balance="120",events=es,stop=("fun",),methods=("full_payment",)),flows); self.assertLessEqual(len(r.payment_plan.spending_changes),3)
  protected=d.decide(ctx(balance="100",events=(es[0],),stop=("fun",),protect=("fun",)),flows[:1]); self.assertEqual("not_recommended",protected.recommended_payment_method)
 def test_ranking_cost_start_count_and_id(self):
  d=DecisionEngine(ForecastEngine({})); a=opt("b","20",4,S,total="80"); b=opt("a","40",2,S,total="80"); c=opt("c","30",3,S,total="90")
  r=d.decide(ctx(balance="100",methods=("installments",),options=(a,b,c),maxmonths=4)); self.assertEqual("a",r.payment_plan.payment_option_id)
if __name__=="__main__": unittest.main()
