#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Actions finalize 通知脚本：统计 297 城成败 → 发钉钉 + 邮件
使用环境变量（GitHub Secrets 注入，未配置则跳过对应渠道）：
  DINGTALK_WEBHOOK  钉钉群机器人 webhook URL（含 access_token）
  DINGTALK_SECRET   钉钉加签密钥（机器人用了"加签"安全设置时才填，否则留空）
  SMTP_HOST         邮箱 SMTP 服务器（如 smtp.qq.com）
  SMTP_PORT         端口（465 用 SSL，587 用 STARTTLS）
  SMTP_USER         发件邮箱账号
  SMTP_PASS         授权码（QQ 邮箱/163 等在设置里生成，不是登录密码）
  MAIL_TO           收件邮箱（多个用英文逗号分隔）
"""
import csv, glob, os, sys, json, time, hmac, hashlib, base64
import urllib.request, urllib.parse

def load_need():
    """cities.csv 里应计算的全部城市 file -> 中文名"""
    need = {}
    with open("cities.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            need[r["file"]] = r["name"]
    return need

def load_done():
    """data/ 下实际产出 _forecast.csv 的城市 file 集合"""
    done = set()
    for p in glob.glob("data/**/*_forecast.csv", recursive=True):
        b = os.path.basename(p)
        if b.endswith("_forecast.csv"):
            done.add(b[: -len("_forecast.csv")])
    return done

def build_text(need, done):
    failed = [k for k in need if k not in done]
    ok = len(need) - len(failed)
    lines = ["### 自算天气预报 · 每日更新完成",
             f"**总城市 {len(need)} | 成功 {ok} | 失败 {len(failed)}**"]
    if failed:
        names = ", ".join(f"{need[k]}({k})" for k in failed)
        lines.append(f"**失败城市（可手动重跑）:**\n\n{names}")
    else:
        lines.append("全部城市更新成功 ✅")
    return "\n\n".join(lines), failed

def dingtalk(webhook, secret, text, title):
    ts = str(round(time.time() * 1000))
    url = webhook
    if secret:
        string_to_sign = "%s\n%s" % (ts, secret)
        h = hmac.new(secret.encode(), string_to_sign.encode(), hashlib.sha256).digest()
        url += "&timestamp=%s&sign=%s" % (ts, urllib.parse.quote(base64.b64encode(h)))
    payload = {"msgtype": "markdown", "markdown": {"title": title, "text": text}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10)

def mail(host, port, user, pw, to, subject, body):
    from email.mime.text import MIMEText
    import smtplib
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    if int(port) == 465:
        s = smtplib.SMTP_SSL(host, int(port), timeout=15)
    else:
        s = smtplib.SMTP(host, int(port), timeout=15)
        s.starttls()
    s.login(user, pw)
    s.sendmail(user, to.split(","), msg.as_string())
    s.quit()

def main():
    need, done = load_need(), load_done()
    text, failed = build_text(need, done)
    ok = len(need) - len(failed)
    subject = "天气预报更新：%d 城成功 %d 失败" % (ok, len(failed))

    wh = os.environ.get("DINGTALK_WEBHOOK", "").strip()
    if wh:
        try:
            dingtalk(wh, os.environ.get("DINGTALK_SECRET", ""), text, subject)
            print("[notify] 钉钉已发送")
        except Exception as e:
            print("[notify] 钉钉发送失败:", e)
    else:
        print("[notify] 未配置 DINGTALK_WEBHOOK，跳过钉钉通知")

    if os.environ.get("SMTP_HOST", "").strip():
        try:
            mail(os.environ["SMTP_HOST"], os.environ.get("SMTP_PORT", "465"),
                 os.environ["SMTP_USER"], os.environ["SMTP_PASS"],
                 os.environ.get("MAIL_TO", os.environ["SMTP_USER"]),
                 subject, text)
            print("[notify] 邮件已发送")
        except Exception as e:
            print("[notify] 邮件发送失败:", e)
    else:
        print("[notify] 未配置 SMTP_*，跳过邮件通知")

if __name__ == "__main__":
    main()
