"""Synthetic Turkish corpus and QA set generator.

The corpus imitates three kinds of institutional knowledge base: municipal service guides, company HR
policies and appliance manuals. Every entity (5 fictional municipalities, 4 fictional companies,
6 fictional appliance models) gets the *same* document types with *different* values, so documents come in
families of near-duplicates — the realistic hard case where retrieval must find the right entity, not just
the right topic.

Every fact is written once into one document, and its answer span is unique in that document, so relevance
can be decided mechanically: a chunk is relevant to a question when it comes from the gold document and
contains the answer span. Each fact yields three questions:

* ``canonical`` — formal Turkish, close to the document wording;
* ``paraphrase`` — different vocabulary and inflection (``güvence bedeli`` → ``depozito``);
* ``ascii`` — the canonical question lowercased, without Turkish letters or punctuation, as people type
  into search boxes (``yesilvadi belediyesinde ...``).

Plus unanswerable questions about topics no document covers. Names are invented; any resemblance to a
real institution is unintended. Same seed → byte-identical output (tested).
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ragtr.text.turkish import fold_diacritics, tr_lower

SEED = 2026


@dataclass(frozen=True, slots=True)
class QA:
    id: str
    question: str
    variant: str  # canonical | paraphrase | ascii | unanswerable
    answerable: bool
    document_id: str | None = None
    answer: str | None = None  # reference answer
    span: str | None = None  # exact, unique substring of the gold document
    must: list[str] = field(default_factory=list)  # normalised substrings a correct answer contains
    fact: str | None = None


@dataclass(frozen=True, slots=True)
class Fact:
    key: str
    span: str
    answer: str
    must: list[str]
    canonical: str
    paraphrase: str


@dataclass(frozen=True, slots=True)
class SynthDoc:
    id: str
    title: str
    text: str
    facts: list[Fact]


@dataclass(frozen=True, slots=True)
class Entity:
    slug: str
    name: str  # "Yeşilvadi"
    loc: str  # "Yeşilvadi'de"
    org: str  # "Yeşilvadi Belediyesi"
    org_loc: str  # "Yeşilvadi Belediyesi'nde"
    gen: str = ""  # genitive, used by appliances: "K-200 buzdolabının"


MUNICIPALITIES = [
    Entity("yesilvadi", "Yeşilvadi", "Yeşilvadi'de", "Yeşilvadi Belediyesi", "Yeşilvadi Belediyesi'nde"),
    Entity("akcaburc", "Akçaburç", "Akçaburç'ta", "Akçaburç Belediyesi", "Akçaburç Belediyesi'nde"),
    Entity("isikliova", "Işıklıova", "Işıklıova'da", "Işıklıova Belediyesi", "Işıklıova Belediyesi'nde"),
    Entity("ikizpinar", "İkizpınar", "İkizpınar'da", "İkizpınar Belediyesi", "İkizpınar Belediyesi'nde"),
    Entity("camliyaka", "Çamlıyaka", "Çamlıyaka'da", "Çamlıyaka Belediyesi", "Çamlıyaka Belediyesi'nde"),
]
COMPANIES = [
    Entity("defne", "Defne Yazılım", "Defne Yazılım'da", "Defne Yazılım A.Ş.", "Defne Yazılım'da"),
    Entity("maviliman", "Mavi Liman Lojistik", "Mavi Liman Lojistik'te", "Mavi Liman Lojistik Ltd. Şti.",
           "Mavi Liman Lojistik'te"),
    Entity("kuzeyruzgari", "Kuzeyrüzgârı Enerji", "Kuzeyrüzgârı Enerji'de", "Kuzeyrüzgârı Enerji A.Ş.",
           "Kuzeyrüzgârı Enerji'de"),
    Entity("isilti", "Işıltı Gıda", "Işıltı Gıda'da", "Işıltı Gıda San. ve Tic. A.Ş.", "Işıltı Gıda'da"),
]  # fmt: skip
APPLIANCES = [
    Entity("k200", "Ayaz K-200 buzdolabı", "Ayaz K-200 buzdolabında", "Ayaz K-200 Buzdolabı",
           "Ayaz K-200 buzdolabında", "Ayaz K-200 buzdolabının"),
    Entity("c310", "Ayaz Ç-310 çamaşır makinesi", "Ayaz Ç-310 çamaşır makinesinde", "Ayaz Ç-310 Çamaşır Makinesi",
           "Ayaz Ç-310 çamaşır makinesinde", "Ayaz Ç-310 çamaşır makinesinin"),
    Entity("b120", "Ayaz B-120 bulaşık makinesi", "Ayaz B-120 bulaşık makinesinde", "Ayaz B-120 Bulaşık Makinesi",
           "Ayaz B-120 bulaşık makinesinde", "Ayaz B-120 bulaşık makinesinin"),
    Entity("f45", "Ayaz F-45 ankastre fırın", "Ayaz F-45 fırında", "Ayaz F-45 Ankastre Fırın",
           "Ayaz F-45 fırında", "Ayaz F-45 fırının"),
    Entity("s90", "Ayaz S-90 dikey süpürge", "Ayaz S-90 süpürgede", "Ayaz S-90 Dikey Süpürge",
           "Ayaz S-90 süpürgede", "Ayaz S-90 süpürgenin"),
    Entity("i75", "Ayaz İ-75 ısı pompası", "Ayaz İ-75 ısı pompasında", "Ayaz İ-75 Isı Pompası",
           "Ayaz İ-75 ısı pompasında", "Ayaz İ-75 ısı pompasının"),
]  # fmt: skip

# Boilerplate that appears in every document of a family: realistic noise, and it carries no facts.
FILLER = {
    "municipal": [
        "Başvurular e-Devlet üzerinden de yapılabilir; ancak bazı belgelerin aslı birim tarafından görülmelidir.",
        "Vatandaşlarımızın işlemlerini hızlandırmak için belgelerin eksiksiz getirilmesi önemlidir.",
        "Engelli ve yaşlı vatandaşlarımız için öncelikli hizmet masası bulunmaktadır.",
        "Ayrıntılı bilgi için belediyenin çağrı merkezine veya internet sitesine başvurulabilir.",
        "Mevzuatta yapılan değişiklikler bu rehbere en kısa sürede yansıtılır.",
        "İşlemler sırasında örn. vekâletname gibi ek belgeler de istenebilir.",
        "Belediye meclisinin aldığı kararlar doğrultusunda tarifeler her yıl yeniden belirlenir.",
        "Şikâyet ve önerileriniz için Beyaz Masa birimine yazılı olarak başvurabilirsiniz.",
    ],
    "hr": [
        "Bu politika tüm tam zamanlı çalışanlar için geçerlidir; yarı zamanlı çalışanlar için oranlama yapılır.",
        "Politikada yer almayan durumlarda İnsan Kaynakları Direktörlüğü'nün yorumu esas alınır.",
        "Çalışanlar, yöneticileriyle birlikte planlama yaparak ekip içi iş sürekliliğini korumalıdır.",
        "Talepler insan kaynakları portalı üzerinden elektronik olarak iletilir ve kayıt altına alınır.",
        "Politika yılda bir kez gözden geçirilir; değişiklikler tüm çalışanlara e-posta ile duyurulur.",
        "Yasal düzenlemelerin bu politikadan daha lehe olduğu durumlarda yasal düzenleme uygulanır.",
        "Yöneticiler, talepleri makul süre içinde yanıtlamakla yükümlüdür.",
    ],
    "manual": [
        "Cihazı kullanmadan önce bu kılavuzu dikkatlice okuyunuz ve ileride başvurmak üzere saklayınız.",
        "Bakım ve temizlik işlemlerinden önce cihazın fişini prizden çekiniz.",
        "Cihazın üzerinde yer alan etiketleri çıkarmayınız; bunlar servis işlemleri için gereklidir.",
        "Yetkisiz kişilerce yapılan müdahaleler garanti kapsamını sona erdirir.",
        "Ambalaj malzemelerini çocukların ulaşamayacağı bir yerde saklayınız.",
        "Cihazı doğrudan güneş ışığına ve ısı kaynaklarına yakın yerlere yerleştirmeyiniz.",
        "Daha fazla bilgi için yetkili servis noktalarına veya müşteri hizmetlerine başvurunuz.",
    ],
}


def tl(amount: int) -> str:
    """Turkish money format: 1450 → '1.450 TL'."""
    return f"{amount:,}".replace(",", ".") + " TL"


def _fill(rng: random.Random, family: str, n: int = 1) -> str:
    return " ".join(rng.sample(FILLER[family], n))


def _hour(rng: random.Random, choices: list[str]) -> str:
    return rng.choice(choices)


# --------------------------------------------------------------------------------------- municipal


def water(e: Entity, rng: random.Random) -> SynthDoc:
    floor, days = rng.sample(range(1, 6), 2)
    deposit = rng.randrange(900, 2600, 50)
    opens, closes = _hour(rng, ["08.00", "08.30", "09.00"]), _hour(rng, ["16.30", "17.00", "17.30"])
    text = f"""# {e.org} Su Aboneliği Rehberi

{e.org} Su ve Kanalizasyon İşleri Müdürlüğü, ilçe sınırları içindeki konut ve iş yerlerine içme suyu aboneliği hizmeti verir. Bu rehber; başvuru, ücretler ve sayaç işlemleri hakkında bilgi verir. {_fill(rng, "municipal")}

## Başvuru

Abonelik başvurusu, belediye hizmet binasının {floor}. katındaki abone işlemleri biriminde yapılır. Birim hafta içi her gün {opens} ile {closes} saatleri arasında açıktır. Başvuru sırasında kimlik belgesi, tapu veya kira sözleşmesi ve yapı kullanma izin belgesinin bir örneği istenir. {_fill(rng, "municipal")}

## Ücretler

Yeni konut aboneliklerinde {tl(deposit)} güvence bedeli alınır. Güvence bedeli, abonelik sona erdiğinde borç bulunmaması hâlinde güncel tutar üzerinden iade edilir. {_fill(rng, "municipal", 2)}

## Sayaç Takılması

Başvurunun onaylanmasından sonra sayaç en geç {days} iş günü içinde takılır. Sayacın takılacağı yer, okuma görevlisinin kolayca ulaşabileceği şekilde hazırlanmalıdır. {_fill(rng, "municipal")}
"""
    return SynthDoc(f"{e.slug}-su-aboneligi", f"{e.org} Su Aboneliği Rehberi", text, [
        Fact("deposit", tl(deposit), tl(deposit), [f"{deposit:,}".replace(",", ".")],
             f"{e.org_loc} yeni konut su aboneliği için güvence bedeli ne kadardır?",
             f"{e.loc} eve su bağlatırken kaç lira depozito ödenir?"),
        Fact("meter_days", f"{days} iş günü", f"{days} iş günü", [str(days), "gün"],
             f"{e.org_loc} su sayacı başvuru onaylandıktan sonra kaç iş günü içinde takılır?",
             f"{e.loc} su saatim ne kadar sürede bağlanır?"),
        Fact("floor", f"{floor}. kat", f"{floor}. kat", [f"{floor}. kat"],
             f"{e.org_loc} su aboneliği başvurusu hizmet binasının hangi katında yapılır?",
             f"{e.loc} su abonesi olmak için belediye binasında nereye gitmeliyim?"),
        Fact("hours", f"{opens} ile {closes}", f"{opens}-{closes}", [opens, closes],
             f"{e.org_loc} abone işlemleri birimi hangi saatler arasında açıktır?",
             f"{e.loc} su abonelik masası sabah kaçta açılıp akşam kaçta kapanıyor?"),
    ])  # fmt: skip


def building_permit(e: Entity, rng: random.Random) -> SynthDoc:
    review, copies, floor = rng.sample(range(2, 9), 3)
    review *= 5
    fee = rng.randrange(18, 60)
    text = f"""# {e.org} Yapı Ruhsatı Başvuru Rehberi

Yeni bina, ek kat veya esaslı tadilat yapmak isteyen mülk sahipleri {e.org} İmar ve Şehircilik Müdürlüğü'nden yapı ruhsatı almak zorundadır. Ruhsatsız yapılara 3194 sayılı İmar Kanunu uyarınca yaptırım uygulanır. {_fill(rng, "municipal")}

## Gerekli Belgeler

Başvuru dosyasında tapu kaydı, aplikasyon krokisi, zemin etüdü raporu ve mimari, statik, elektrik ve mekanik projeler bulunur. Projeler {copies} takım hâlinde ıslak imzalı olarak teslim edilir. Dosyalar müdürlüğün bulunduğu ek hizmet binasının {floor}. katındaki evrak kabul bürosuna verilir. {_fill(rng, "municipal")}

## İnceleme Süreci

Eksiksiz dosyalar en geç {review} gün içinde incelenir ve sonuç başvuru sahibine yazılı olarak bildirilir. Eksik belge tespit edilirse süre, eksikliğin giderildiği tarihten itibaren yeniden başlar. {_fill(rng, "municipal")}

## Harçlar

Konut yapılarında ruhsat harcı metrekare başına {fee} TL olarak uygulanır. Ticari yapılar için ayrı tarife geçerlidir; ayrıntılar için bkz. belediye gelir tarifesi. {_fill(rng, "municipal")}
"""
    return SynthDoc(f"{e.slug}-yapi-ruhsati", f"{e.org} Yapı Ruhsatı Başvuru Rehberi", text, [
        Fact("copies", f"{copies} takım", f"{copies} takım", [str(copies)],
             f"{e.org_loc} yapı ruhsatı için projeler kaç takım hâlinde teslim edilir?",
             f"{e.loc} inşaat izni alırken proje dosyasından kaç kopya vermem gerekiyor?"),
        Fact("review", f"{review} gün içinde", f"{review} gün", [str(review), "gün"],
             f"{e.org_loc} eksiksiz yapı ruhsatı dosyaları kaç gün içinde incelenir?",
             f"{e.loc} ruhsat başvurum ne kadar sürede sonuçlanır?"),
        Fact("fee", f"metrekare başına {fee} TL", f"metrekare başına {fee} TL", [str(fee)],
             f"{e.org_loc} konut yapılarında ruhsat harcı metrekare başına ne kadardır?",
             f"{e.loc} ev inşaatı için m² başına kaç lira harç ödenir?"),
    ])  # fmt: skip


def marriage(e: Entity, rng: random.Random) -> SynthDoc:
    weekday = rng.randrange(300, 900, 50)
    weekend = weekday + rng.randrange(200, 700, 50)
    lead = rng.choice([10, 15, 20, 30])
    capacity = rng.randrange(80, 260, 20)
    hall = rng.choice(["Gül", "Lale", "Menekşe", "Zambak", "Papatya"]) + " Nikâh Salonu"
    text = f"""# {e.org} Nikâh İşlemleri

{e.org} Evlendirme Memurluğu, nikâh başvurularını kabul eder ve nikâh törenlerini gerçekleştirir. Törenler belediyenin {hall}'nda yapılır; salon {capacity} kişi kapasitelidir. {_fill(rng, "municipal")}

## Başvuru Zamanı

Çiftlerin nikâh tarihinden en az {lead} gün önce evlendirme memurluğuna başvurması gerekir. Başvuruda nüfus cüzdanı, sağlık raporu ve vesikalık fotoğraf istenir. Yabancı uyruklu adaylar için ek belgeler gereklidir; ayrıntılar için bkz. Dışişleri mevzuatı. {_fill(rng, "municipal")}

## Ücretler

Hafta içi yapılan nikâhlar için {tl(weekday)}, hafta sonu ve resmî tatil günlerinde yapılan nikâhlar için {tl(weekend)} ücret alınır. Ücrete evlilik cüzdanı bedeli dâhil değildir. {_fill(rng, "municipal")}
"""
    return SynthDoc(f"{e.slug}-nikah-islemleri", f"{e.org} Nikâh İşlemleri", text, [
        Fact("lead", f"en az {lead} gün önce", f"en az {lead} gün önce", [str(lead), "gün"],
             f"{e.org_loc} nikâh başvurusu nikâh tarihinden en az kaç gün önce yapılmalıdır?",
             f"{e.loc} evlenmek için belediyeye ne kadar önceden müracaat etmeliyiz?"),
        Fact("weekend_fee", tl(weekend), tl(weekend), [f"{weekend:,}".replace(",", ".")],
             f"{e.org_loc} hafta sonu yapılan nikâhlar için ne kadar ücret alınır?",
             f"{e.loc} cumartesi günü evlenirsek nikâh kaç lira tutar?"),
        Fact("capacity", f"{capacity} kişi", f"{capacity} kişi", [str(capacity)],
             f"{e.org} {hall} kaç kişi kapasitelidir?",
             f"{e.loc} nikâh salonuna en fazla kaç davetli sığar?"),
    ])  # fmt: skip


def waste(e: Entity, rng: random.Random) -> SynthDoc:
    days = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi"]
    d1, d2 = sorted(rng.sample(range(6), 2))
    bulky_limit = rng.choice([1, 2, 3])
    start = _hour(rng, ["06.00", "07.00", "22.00", "23.00"])
    phone = f"0 (2{rng.randrange(10, 99)}) 555 {rng.randrange(10, 99)} {rng.randrange(10, 99)}"
    text = f"""# {e.org} Katı Atık Toplama Programı

{e.org} Temizlik İşleri Müdürlüğü evsel atıkları düzenli olarak toplar ve ayrıştırma tesisine ulaştırır. Atıkların kapalı poşetlerde, konteyner içine bırakılması gerekir. {_fill(rng, "municipal")}

## Toplama Günleri

Evsel atıklar her gün toplanır. Geri dönüştürülebilir atıklar ise haftada iki kez, {days[d1]} ve {days[d2]} günleri toplanır. Toplama araçları sahaya saat {start} itibarıyla çıkar. {_fill(rng, "municipal")}

## Hacimli Atıklar

Eski mobilya, beyaz eşya ve benzeri hacimli atıklar için Alo Temizlik hattı {phone} aranarak randevu alınır. Her konut için ayda {bulky_limit} kez ücretsiz hacimli atık alımı yapılır; fazlası ücrete tabidir. {_fill(rng, "municipal")}
"""
    return SynthDoc(f"{e.slug}-atik-toplama", f"{e.org} Katı Atık Toplama Programı", text, [
        Fact("recycling_days", f"{days[d1]} ve {days[d2]}", f"{days[d1]} ve {days[d2]}",
             [tr_lower(days[d1]), tr_lower(days[d2])],
             f"{e.org_loc} geri dönüştürülebilir atıklar hangi günler toplanır?",
             f"{e.loc} plastik ve kâğıt çöpleri hangi günlerde alıyorlar?"),
        Fact("bulky_limit", f"ayda {bulky_limit} kez", f"ayda {bulky_limit} kez", [str(bulky_limit)],
             f"{e.org_loc} hacimli atıklar ayda kaç kez ücretsiz alınır?",
             f"{e.loc} eski koltuğumu belediye ücretsiz olarak ayda kaç defa götürür?"),
        Fact("phone", phone, phone, [phone[-5:]],
             f"{e.org} Alo Temizlik hattının telefon numarası nedir?",
             f"{e.loc} eski buzdolabını aldırmak için hangi numarayı aramalıyım?"),
    ])  # fmt: skip


def market(e: Entity, rng: random.Random) -> SynthDoc:
    days = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
    day = rng.choice(days)
    place = rng.choice(["Cumhuriyet", "Atatürk", "Yıldız", "Fatih", "Gazi", "Kemerli"]) + " Mahallesi"
    fee = rng.randrange(60, 250, 10)
    opens = _hour(rng, ["07.00", "07.30", "08.00"])
    text = f"""# {e.org} Semt Pazarları

{e.org} Zabıta Müdürlüğü, ilçedeki semt pazarlarının düzenini ve denetimini sağlar. Pazar yerlerinde fiyat etiketi kullanılması zorunludur. {_fill(rng, "municipal")}

## Pazar Günü ve Yeri

İlçenin en büyük semt pazarı her hafta {day} günü {place}'ndeki kapalı pazar alanında kurulur. Pazar alanı saat {opens}'de alışverişe açılır. {_fill(rng, "municipal")}

## Tezgâh İşlemleri

Pazarcı esnafından tezgâh başına günlük {fee} TL işgal harcı alınır. Tezgâh tahsisleri her yıl Ocak ayında kura ile yenilenir. Hijyen kurallarına uymayan esnafa idari para cezası uygulanır. {_fill(rng, "municipal")}
"""
    return SynthDoc(f"{e.slug}-semt-pazarlari", f"{e.org} Semt Pazarları", text, [
        Fact("day", f"{day} günü", f"{day} günü {place}", [tr_lower(day)],
             f"{e.org_loc} en büyük semt pazarı hangi gün kurulur?",
             f"{e.loc} haftalık büyük pazar ne zaman açılıyor?"),
        Fact("fee", f"günlük {fee} TL", f"tezgâh başına günlük {fee} TL", [str(fee)],
             f"{e.org_loc} pazarcı esnafından tezgâh başına günlük ne kadar işgal harcı alınır?",
             f"{e.loc} pazarda tezgâh açmak bir günlüğüne kaç liraya mal olur?"),
        Fact("place", place, place, [tr_lower(place.split()[0])],
             f"{e.org_loc} en büyük semt pazarı hangi mahallede kurulur?",
             f"{e.loc} büyük kapalı pazar alanı nerede?"),
    ])  # fmt: skip


# --------------------------------------------------------------------------------------------- HR


def annual_leave(e: Entity, rng: random.Random) -> SynthDoc:
    first, senior, carry, notice = (
        14 + rng.randrange(0, 4),
        20 + rng.randrange(0, 6),
        *rng.sample(range(3, 11), 2),
    )
    text = f"""# {e.org} Yıllık İzin Politikası

Bu politika, {e.org} çalışanlarının yıllık ücretli izin haklarını ve izin kullanım usullerini düzenler. {_fill(rng, "hr")}

## İzin Süreleri

Bir yılını dolduran çalışanlara yılda {first} iş günü ücretli izin verilir. Kıdemi beş yılı aşan çalışanların yıllık izin süresi {senior} iş günüdür. Resmî tatiller izin süresinden sayılmaz. {_fill(rng, "hr")}

## İzin Talebi

İzin talepleri, iznin başlayacağı tarihten en az {notice} gün önce yöneticiye iletilmelidir. Acil durumlarda bu süre şartı aranmaz. {_fill(rng, "hr")}

## Devreden İzinler

Kullanılmayan izinlerden en fazla {carry} gün bir sonraki yıla devredebilir. Devreden günler Mart ayı sonuna kadar kullanılmazsa yanar. {_fill(rng, "hr")}
"""
    return SynthDoc(f"{e.slug}-yillik-izin", f"{e.org} Yıllık İzin Politikası", text, [
        Fact("first_year", f"yılda {first} iş günü", f"{first} iş günü", [str(first)],
             f"{e.org_loc} bir yılını dolduran çalışana kaç iş günü yıllık izin verilir?",
             f"{e.loc} işe girdikten bir yıl sonra kaç gün tatil hakkım oluyor?"),
        Fact("senior", f"{senior} iş günüdür", f"{senior} iş günü", [str(senior)],
             f"{e.org_loc} kıdemi beş yılı aşan çalışanların yıllık izin süresi kaç gündür?",
             f"{e.loc} beş yıldan uzun süredir çalışanlar yılda ne kadar izin kullanabiliyor?"),
        Fact("notice", f"en az {notice} gün önce", f"en az {notice} gün önce", [str(notice), "gün"],
             f"{e.org_loc} izin talepleri iznin başlamasından en az kaç gün önce iletilmelidir?",
             f"{e.loc} tatile çıkmadan ne kadar önce yöneticime haber vermeliyim?"),
        Fact("carry", f"en fazla {carry} gün", f"en fazla {carry} gün", [str(carry)],
             f"{e.org_loc} kullanılmayan izinlerden en fazla kaç gün sonraki yıla devreder?",
             f"{e.loc} bu yıl kullanmadığım izin günlerinin ne kadarını seneye aktarabilirim?"),
    ])  # fmt: skip


def remote_work(e: Entity, rng: random.Random) -> SynthDoc:
    days = rng.choice([1, 2, 3])
    allowance = rng.randrange(250, 800, 50)
    core_start, core_end = rng.choice(["10.00", "10.30", "11.00"]), rng.choice(["15.00", "15.30", "16.00"])
    text = f"""# {e.org} Uzaktan Çalışma Politikası

{e.org}, görev tanımı uygun olan çalışanlarına hibrit çalışma imkânı sunar. Uzaktan çalışma bir hak değil, yöneticinin onayına bağlı bir esnekliktir. {_fill(rng, "hr")}

## Uzaktan Çalışma Günleri

Çalışanlar haftada en fazla {days} gün uzaktan çalışabilir. Uzaktan çalışma günleri ekip takvimine önceden işlenir. Tüm çalışanlar, çalışma yerinden bağımsız olarak {core_start} ile {core_end} arasındaki ortak çalışma saatlerinde ulaşılabilir olmalıdır. {_fill(rng, "hr")}

## Giderlerin Karşılanması

Uzaktan çalışan personele internet ve elektrik giderleri için aylık {tl(allowance)} destek ödenir. Ödeme maaşla birlikte yapılır. Ev ofis mobilyası talepleri ayrıca değerlendirilir. {_fill(rng, "hr")}
"""
    return SynthDoc(f"{e.slug}-uzaktan-calisma", f"{e.org} Uzaktan Çalışma Politikası", text, [
        Fact("days", f"haftada en fazla {days} gün", f"haftada en fazla {days} gün", [str(days), "gün"],
             f"{e.org_loc} çalışanlar haftada en fazla kaç gün uzaktan çalışabilir?",
             f"{e.loc} evden çalışmaya haftanın kaç günü izin veriliyor?"),
        Fact("allowance", f"aylık {tl(allowance)}", f"aylık {tl(allowance)}", [str(allowance)],
             f"{e.org_loc} uzaktan çalışan personele aylık ne kadar gider desteği ödenir?",
             f"{e.loc} evden çalışırken internet faturası için şirket ayda kaç lira veriyor?"),
        Fact("core_hours", f"{core_start} ile {core_end}", f"{core_start}-{core_end}", [core_start, core_end],
             f"{e.org_loc} ortak çalışma saatleri nelerdir?",
             f"{e.loc} evden çalışsam bile hangi saatlerde mutlaka erişilebilir olmam gerekir?"),
    ])  # fmt: skip


def expenses(e: Entity, rng: random.Random) -> SynthDoc:
    meal = rng.randrange(300, 700, 25)
    hotel = rng.randrange(2000, 5000, 250)
    deadline = rng.choice([15, 20, 30, 45])
    text = f"""# {e.org} Masraf ve Harcırah Yönergesi

Bu yönerge, {e.org} çalışanlarının iş seyahatleri sırasında yaptıkları harcamaların karşılanma esaslarını belirler. {_fill(rng, "hr")}

## Günlük Harcırah

Şehir dışı görevlendirmelerde çalışanlara yemek giderleri için günlük {tl(meal)} harcırah ödenir. Konaklama giderleri ise gecelik {tl(hotel)} üst sınırına kadar karşılanır. Bu tutarı aşan harcamalar için genel müdür yardımcısının onayı gerekir. {_fill(rng, "hr")}

## Masraf Beyanı

Masraf formları, harcamanın yapıldığı tarihten itibaren {deadline} gün içinde faturalarıyla birlikte muhasebe birimine teslim edilmelidir. Süresi içinde beyan edilmeyen masraflar ödenmez. {_fill(rng, "hr")}
"""
    return SynthDoc(f"{e.slug}-masraf-harcirah", f"{e.org} Masraf ve Harcırah Yönergesi", text, [
        Fact("meal", f"günlük {tl(meal)}", f"günlük {tl(meal)}", [str(meal)],
             f"{e.org_loc} şehir dışı görevlendirmelerde günlük yemek harcırahı ne kadardır?",
             f"{e.loc} iş seyahatinde yemek için günde kaç lira veriliyor?"),
        Fact("hotel", f"gecelik {tl(hotel)}", f"gecelik {tl(hotel)}", [f"{hotel:,}".replace(",", ".")],
             f"{e.org_loc} konaklama giderleri gecelik en fazla ne kadara kadar karşılanır?",
             f"{e.loc} seyahatte otel için bir gece en çok kaç liralık oda tutabilirim?"),
        Fact("deadline", f"{deadline} gün içinde", f"{deadline} gün içinde", [str(deadline), "gün"],
             f"{e.org_loc} masraf formları harcamadan sonra kaç gün içinde teslim edilmelidir?",
             f"{e.loc} fişlerimi muhasebeye vermek için ne kadar vaktim var?"),
    ])  # fmt: skip


def onboarding(e: Entity, rng: random.Random) -> SynthDoc:
    orientation = rng.choice([2, 3, 4, 5])
    mentor = rng.choice([60, 90, 120])
    probation = rng.choice([1, 2])
    text = f"""# {e.org} İşe Başlangıç Rehberi

{e.org} ailesine yeni katılan çalışanların uyum sürecini kolaylaştırmak amacıyla hazırlanmıştır. {_fill(rng, "hr")}

## Oryantasyon Programı

Yeni çalışanlar ilk haftalarında {orientation} gün süren oryantasyon programına katılır. Programda şirket kültürü, bilgi güvenliği ve iş sağlığı ve güvenliği eğitimleri verilir. {_fill(rng, "hr")}

## Mentorluk

Her yeni çalışana ilk {mentor} gün boyunca deneyimli bir mentor atanır. Mentor, haftalık görüşmelerle yeni çalışanın sorularını yanıtlar. Deneme süresi {probation} aydır; bu süre sonunda yöneticiyle değerlendirme görüşmesi yapılır. {_fill(rng, "hr")}
"""
    return SynthDoc(f"{e.slug}-ise-baslangic", f"{e.org} İşe Başlangıç Rehberi", text, [
        Fact("orientation", f"{orientation} gün süren", f"{orientation} gün", [str(orientation), "gün"],
             f"{e.org_loc} oryantasyon programı kaç gün sürer?",
             f"{e.loc} yeni başlayanların uyum eğitimi ne kadar sürüyor?"),
        Fact("mentor", f"ilk {mentor} gün", f"ilk {mentor} gün", [str(mentor)],
             f"{e.org_loc} yeni çalışana kaç gün boyunca mentor atanır?",
             f"{e.loc} işe yeni girene rehberlik eden kişi ne kadar süre yanında olur?"),
        Fact("probation", f"{probation} aydır", f"{probation} ay", [str(probation), "ay"],
             f"{e.org_loc} deneme süresi kaç aydır?",
             f"{e.loc} yeni işe girenlerin deneme dönemi ne kadar?"),
    ])  # fmt: skip


# ------------------------------------------------------------------------------------- appliances

ERRORS = [
    ("su girişi sağlanamıyor", "musluğun açık olduğunu ve giriş hortumunun bükülmediğini kontrol ediniz"),
    ("tahliye pompası tıkalı", "pompa filtresini temizleyiniz"),
    ("kapı tam kapanmamış", "kapıyı yeniden kapatınız ve conta çevresini kontrol ediniz"),
    ("sıcaklık sensörü arızalı", "yetkili servise başvurunuz"),
    ("motor aşırı ısındı", "cihazı 30 dakika dinlendirip yeniden çalıştırınız"),
    ("şebeke gerilimi düşük", "cihazı ayrı bir prize bağlayınız"),
    ("fan dönmüyor", "fan çevresindeki buzlanmayı ve tozu temizleyiniz"),
    ("filtre doldu", "filtreyi değiştiriniz"),
]


def manual(e: Entity, rng: random.Random) -> SynthDoc:
    energy = rng.choice(["A", "B", "C", "D"])
    power = rng.randrange(80, 2400, 20)
    noise = rng.randrange(38, 72)
    text = f"""# {e.org} Kullanım Kılavuzu

Bu kılavuz, {e.name} modelinin güvenli kurulumu ve verimli kullanımı için hazırlanmıştır. {_fill(rng, "manual")}

## Teknik Özellikler

Cihazın enerji verimliliği sınıfı {energy} olarak belirlenmiştir. Nominal güç tüketimi {power} W, çalışma sırasındaki ses seviyesi {noise} dB'dir. Cihaz 220-240 V, 50 Hz şebeke gerilimiyle çalışır. {_fill(rng, "manual")}

## Kurulum

Cihaz düz ve sağlam bir zemine yerleştirilmeli, havalandırma boşlukları kapatılmamalıdır. İlk çalıştırmadan önce taşıma emniyet parçalarını çıkarınız. {_fill(rng, "manual", 2)}
"""
    return SynthDoc(f"ayaz-{e.slug}-kilavuz", f"{e.org} Kullanım Kılavuzu", text, [
        Fact("energy", f"sınıfı {energy}", f"{energy} sınıfı", [f"{tr_lower(energy)}"],
             f"{e.gen} enerji verimliliği sınıfı nedir?",
             f"{e.name} elektrik tasarrufu açısından hangi sınıfta?"),
        Fact("power", f"{power} W", f"{power} W", [str(power)],
             f"{e.gen} nominal güç tüketimi kaç watttır?",
             f"{e.name} çalışırken ne kadar elektrik çeker?"),
        Fact("noise", f"{noise} dB", f"{noise} dB", [str(noise)],
             f"{e.gen} çalışma sırasındaki ses seviyesi kaç desibeldir?",
             f"{e.name} ne kadar gürültülü çalışıyor?"),
    ])  # fmt: skip


def error_codes(e: Entity, rng: random.Random) -> SynthDoc:
    codes = sorted(rng.sample(range(10, 60), 4))
    picks = rng.sample(ERRORS, 4)
    rows = "\n".join(
        f"- E-{c}: {meaning.capitalize()}. Çözüm: {fix}."
        for c, (meaning, fix) in zip(codes, picks, strict=True)
    )
    text = f"""# {e.org} Arıza Kodları

{e.name} bir arıza algıladığında ekranda E harfiyle başlayan bir kod gösterir. Aşağıdaki listeyi kontrol ettikten sonra sorun devam ederse yetkili servise başvurunuz. {_fill(rng, "manual")}

## Kod Listesi

{rows}

## Sıfırlama

Arıza giderildikten sonra cihazı kapatıp 1 dakika bekleyiniz ve yeniden açınız. Kod tekrar görünürse cihazı kullanmaya devam etmeyiniz. {_fill(rng, "manual")}
"""
    facts = []
    for c, (meaning, _fix) in list(zip(codes, picks, strict=True))[:3]:
        facts.append(
            Fact(f"E-{c}", meaning.capitalize(), meaning, [w for w in tr_lower(meaning).split() if len(w) > 3][:2],
                 f"{e.name} ekranında görülen E-{c} arıza kodu ne anlama gelir?",
                 f"{e.name} E-{c} hatası veriyor, sorun ne?")
        )  # fmt: skip
    return SynthDoc(f"ayaz-{e.slug}-ariza-kodlari", f"{e.org} Arıza Kodları", text, facts)


def warranty(e: Entity, rng: random.Random) -> SynthDoc:
    years = rng.choice([2, 3, 5])
    response = rng.choice([24, 48, 72])
    filter_months = rng.choice([3, 4, 6, 12])
    text = f"""# {e.org} Garanti ve Servis Koşulları

{e.name}, fatura tarihinden itibaren {years} yıl süreyle malzeme ve işçilik hatalarına karşı garanti altındadır. Garanti belgesi ve fatura, servis işlemlerinde ibraz edilmelidir. {_fill(rng, "manual")}

## Servis Hizmeti

Arıza bildirimlerinde yetkili servis en geç {response} saat içinde müşteriyle iletişime geçerek randevu planlar. Garanti kapsamındaki onarımlarda işçilik ve parça ücreti alınmaz. {_fill(rng, "manual")}

## Periyodik Bakım

Cihazın verimli çalışması için filtrenin {filter_months} ayda bir değiştirilmesi önerilir. Orijinal olmayan yedek parça kullanımından doğan arızalar garanti kapsamı dışındadır. {_fill(rng, "manual")}
"""
    return SynthDoc(f"ayaz-{e.slug}-garanti", f"{e.org} Garanti ve Servis Koşulları", text, [
        Fact("years", f"{years} yıl", f"{years} yıl", [str(years), "yıl"],
             f"{e.name} kaç yıl garantilidir?",
             f"{e.name} aldım, garantisi ne zamana kadar sürüyor?"),
        Fact("response", f"{response} saat", f"{response} saat", [str(response)],
             f"{e.name} için yetkili servis arıza bildiriminden sonra en geç kaç saat içinde iletişime geçer?",
             f"{e.name} bozulursa servis beni ne kadar sürede arar?"),
        Fact("filter", f"{filter_months} ayda bir", f"{filter_months} ayda bir", [str(filter_months), "ay"],
             f"{e.gen} filtresi ne sıklıkla değiştirilmelidir?",
             f"{e.name} için yeni filtre almam gereken aralık nedir?"),
    ])  # fmt: skip


DOC_TYPES: list[tuple[list[Entity], Callable[[Entity, random.Random], SynthDoc]]] = [
    (MUNICIPALITIES, water),
    (MUNICIPALITIES, building_permit),
    (MUNICIPALITIES, marriage),
    (MUNICIPALITIES, waste),
    (MUNICIPALITIES, market),
    (COMPANIES, annual_leave),
    (COMPANIES, remote_work),
    (COMPANIES, expenses),
    (COMPANIES, onboarding),
    (APPLIANCES, manual),
    (APPLIANCES, error_codes),
    (APPLIANCES, warranty),
]

# Topics no document covers. The entity is real in the corpus, so retrieval returns plausible but wrong
# documents: the generator has to notice that the context does not answer the question.
UNANSWERABLE: list[tuple[list[Entity], str]] = [
    (MUNICIPALITIES, "{org_loc} otopark aylık abonman ücreti ne kadardır?"),
    (MUNICIPALITIES, "{org_loc} kreş başvuruları hangi ayda başlar?"),
    (MUNICIPALITIES, "{org_loc} ruhsatsız köpek sahiplerine ne kadar ceza uygulanır?"),
    (MUNICIPALITIES, "{org} meclis toplantıları hangi gün yapılır?"),
    (COMPANIES, "{org_loc} doğum izni kaç haftadır?"),
    (COMPANIES, "{org_loc} yemek kartına ayda ne kadar yükleme yapılır?"),
    (COMPANIES, "{org_loc} özel sağlık sigortası aile bireylerini kapsıyor mu?"),
    (APPLIANCES, "{gen} ağırlığı kaç kilogramdır?"),
    (APPLIANCES, "{gen} fiyatı ne kadardır?"),
]


def _ascii(question: str) -> str:
    """How people type into search boxes: lowercase, no Turkish letters, no apostrophes or punctuation."""
    q = fold_diacritics(tr_lower(question))
    q = re.sub(r"['’]", "", q)
    return re.sub(r"[?.!,]", "", q).strip()


def _unique(text: str, span: str) -> bool:
    return text.count(span) == 1


def generate(seed: int = SEED) -> tuple[list[SynthDoc], list[QA]]:
    rng = random.Random(seed)
    docs: list[SynthDoc] = []
    for entities, make in DOC_TYPES:
        docs.extend(make(e, rng) for e in entities)
    qas: list[QA] = []
    for doc in docs:
        for fact in doc.facts:
            if not _unique(doc.text, fact.span):
                raise AssertionError(f"answer span {fact.span!r} is not unique in {doc.id}")
            base = f"{doc.id}:{fact.key}"
            for variant, question in (
                ("canonical", fact.canonical),
                ("paraphrase", fact.paraphrase),
                ("ascii", _ascii(fact.canonical)),
            ):
                qas.append(QA(f"{base}:{variant}", question, variant, True, doc.id, fact.answer, fact.span,
                              [fold_diacritics(tr_lower(m)) for m in fact.must], fact.key))  # fmt: skip
    for i, (entities, template) in enumerate(UNANSWERABLE):
        for e in entities:
            q = template.format(org=e.org, org_loc=e.org_loc, gen=e.gen)
            qas.append(QA(f"unanswerable:{i}:{e.slug}", q, "unanswerable", False))
    return docs, qas


def write(out_dir: Path, seed: int = SEED) -> tuple[int, int]:
    docs, qas = generate(seed)
    corpus = out_dir / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for old in corpus.glob("*.md"):
        old.unlink()
    for doc in docs:
        (corpus / f"{doc.id}.md").write_text(doc.text, encoding="utf-8", newline="\n")
    with (out_dir / "qa.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for qa in qas:
            fh.write(json.dumps(asdict(qa), ensure_ascii=False) + "\n")
    return len(docs), len(qas)
