# app.py — ARAM 챔피언 대시보드 (아이콘: 챔피언/아이템/스펠/룬)
import os, re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ARAM PS Dashboard", layout="wide")

# ===== 파일 경로(리포 루트) =====
# 변경 ①: 참가자 행 데이터 소스를 '슈퍼라이트2'로 교체
PLAYERS_CSV   = "aram_matches_superlight2.csv"                 # 참가자 행 데이터 (NEW)
ITEM_SUM_CSV  = "item_summary_with_icons.csv"                  # item, icon_url, total_picks, wins, win_rate
CHAMP_CSV     = "champion_icons.csv"                           # champion, champion_icon (또는 icon/icon_url)
RUNE_CSV      = "rune_icons.csv"                               # rune_core, rune_core_icon, rune_sub, rune_sub_icon
SPELL_CSV     = "spell_icons.csv"                              # 스펠이름, 아이콘URL (헤더 자유)

DD_VERSION = "15.16.1"  # Data Dragon 폴백 버전 (필요시 최신으로 교체)

# ===== 유틸 =====
def _exists(path: str) -> bool:
    ok = os.path.exists(path)
    if not ok:
        st.warning(f"파일 없음: `{path}`")
    return ok

def _norm(x: str) -> str:
    return re.sub(r"\s+", "", str(x)).strip().lower()

# ===== 로더 =====
@st.cache_data
def load_players(path: str) -> pd.DataFrame:
    if not _exists(path):
        st.stop()
    # utf-8 우선, 실패시 cp949 폴백
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949")

    # 승패 정리
    if "win_clean" not in df.columns:
        if "win" in df.columns:
            df["win_clean"] = df["win"].astype(str).str.lower().isin(["true","1","t","yes"]).astype(int)
        else:
            df["win_clean"] = 0

    # 아이템 이름 정리
    for c in [c for c in df.columns if re.fullmatch(r"item[0-6]_name", c)]:
        df[c] = df[c].fillna("").astype(str).str.strip()

    # 기본 텍스트 컬럼
    for c in ["spell1","spell2","spell1_name_fix","spell2_name_fix","rune_core","rune_sub","champion"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()
    return df

@st.cache_data
def load_item_summary(path: str) -> pd.DataFrame:
    if not _exists(path):
        return pd.DataFrame()
    g = pd.read_csv(path)
    need = {"item","icon_url","total_picks","wins","win_rate"}
    if not need.issubset(g.columns):
        st.warning(f"`{path}` 헤더 확인 필요 (기대: {sorted(need)}, 실제: {list(g.columns)})")
    if "item" in g.columns:
        g = g[g["item"].astype(str).str.strip() != ""]
    return g

@st.cache_data
def load_champion_icons(path: str) -> dict:
    if not _exists(path):
        return {}
    df = pd.read_csv(path)
    name_col = None
    for c in ["champion","Champion","championName"]:
        if c in df.columns:
            name_col = c
            break
    icon_col = None
    for c in ["champion_icon","icon","icon_url"]:
        if c in df.columns:
            icon_col = c
            break
    if not name_col or not icon_col:
        return {}
    df[name_col] = df[name_col].astype(str).str.strip()
    return dict(zip(df[name_col], df[icon_col]))

@st.cache_data
def load_rune_icons(path: str) -> dict:
    if not _exists(path):
        return {"core": {}, "sub": {}, "shards": {}}
    df = pd.read_csv(path)
    core_map, sub_map, shard_map = {}, {}, {}
    if "rune_core" in df.columns:
        ic = "rune_core_icon" if "rune_core_icon" in df.columns else None
        if ic: core_map = dict(zip(df["rune_core"].astype(str), df[ic].astype(str)))
    if "rune_sub" in df.columns:
        ic = "rune_sub_icon" if "rune_sub_icon" in df.columns else None
        if ic: sub_map = dict(zip(df["rune_sub"].astype(str), df[ic].astype(str)))
    if "rune_shard" in df.columns:
        ic = "rune_shard_icon" if "rune_shard_icon" in df.columns else ("rune_shards_icons" if "rune_shards_icons" in df.columns else None)
        if ic: shard_map = dict(zip(df["rune_shard"].astype(str), df[ic].astype(str)))
    return {"core": core_map, "sub": sub_map, "shards": shard_map}

@st.cache_data
def load_spell_icons(path: str) -> dict:
    """스펠명(여러 형태) -> 아이콘 URL"""
    if not _exists(path):
        return {}
    df = pd.read_csv(path)
    cand_name = [c for c in df.columns if _norm(c) in {"spell","spellname","name","spell1_name_fix","spell2_name_fix","스펠","스펠명"}]
    cand_icon = [c for c in df.columns if _norm(c) in {"icon","icon_url","spelli con","spell_icon"} or "icon" in c.lower()]
    m = {}
    if cand_name and cand_icon:
        name_col, icon_col = cand_name[0], cand_icon[0]
        for n, i in zip(df[name_col].astype(str), df[icon_col].astype(str)):
            m[_norm(n)] = i
            m[str(n).strip()] = i
    else:
        if df.shape[1] >= 2:
            for n, i in zip(df.iloc[:,0].astype(str), df.iloc[:,1].astype(str)):
                m[_norm(n)] = i
                m[str(n).strip()] = i
    return m

# ===== 데이터 로드 =====
df        = load_players(PLAYERS_CSV)
item_sum  = load_item_summary(ITEM_SUM_CSV)
champ_map = load_champion_icons(CHAMP_CSV)
rune_maps = load_rune_icons(RUNE_CSV)
spell_map = load_spell_icons(SPELL_CSV)

ITEM_ICON_MAP = dict(zip(item_sum.get("item", []), item_sum.get("icon_url", [])))

# ===== 사이드바 =====
st.sidebar.title("ARAM PS Controls")
champs = sorted(df["champion"].dropna().unique().tolist()) if "champion" in df.columns else []
selected = st.sidebar.selectbox("Champion", champs, index=0 if champs else None)

# ===== 상단 요약 =====
dsel = df[df["champion"] == selected].copy() if len(champs) else df.head(0).copy()
games = len(dsel)
match_cnt_all = df["matchId"].nunique() if "matchId" in df.columns else len(df)
match_cnt_sel = dsel["matchId"].nunique() if "matchId" in dsel.columns else games
winrate = round(dsel["win_clean"].mean()*100, 2) if games else 0.0
pickrate = round((match_cnt_sel / match_cnt_all * 100), 2) if match_cnt_all else 0.0

c0, ctitle = st.columns([1, 5])
with c0:
    cicon = champ_map.get(selected, "")
    if cicon:
        st.image(cicon, width=64)
with ctitle:
    st.title(f"{selected}")

c1, c2, c3 = st.columns(3)
c1.metric("Games", f"{games}")
c2.metric("Win Rate", f"{winrate}%")
c3.metric("Pick Rate", f"{pickrate}%")

# ===== 아이템 추천 =====
st.subheader("Recommended Items")
if games and any(re.fullmatch(r"item[0-6]_name", c) for c in dsel.columns):
    stacks = []
    for c in [c for c in dsel.columns if re.fullmatch(r"item[0-6]_name", c)]:
        stacks.append(dsel[[c, "win_clean"]].rename(columns={c: "item"}))
    union = pd.concat(stacks, ignore_index=True)
    union = union[union["item"].astype(str).str.strip() != ""]
    top_items = (
        union.groupby("item")
        .agg(total_picks=("item","count"), wins=("win_clean","sum"))
        .reset_index()
    )
    top_items["win_rate"] = (top_items["wins"]/top_items["total_picks"]*100).round(2)
    top_items["icon_url"] = top_items["item"].map(ITEM_ICON_MAP)
    top_items = top_items.sort_values(["total_picks","win_rate"], ascending=[False, False]).head(20)

    st.dataframe(
        top_items[["icon_url","item","total_picks","wins","win_rate"]],
        use_container_width=True,
        column_config={
            "icon_url": st.column_config.ImageColumn("아이콘", width="small"),
            "item": "아이템", "total_picks": "픽수", "wins": "승수", "win_rate": "승률(%)"
        }
    )
else:
    st.info("아이템 이름 컬럼(item0_name~item6_name)이 없어 챔피언별 아이템 집계를 만들 수 없습니다.")

# ===== 스펠 추천 (정규화 아이콘 + '순서 무시' 통합 집계) =====
st.subheader("Recommended Spell Combos")

# 한↔영 별칭 표준화
SPELL_ALIASES = {
    # 한글
    "점멸":"점멸","표식":"표식","눈덩이":"표식","유체화":"유체화","회복":"회복","점화":"점화",
    "정화":"정화","탈진":"탈진","방어막":"방어막","총명":"총명","순간이동":"순간이동",
    # 영문/변형
    "flash":"점멸","mark":"표식","snowball":"표식","ghost":"유체화","haste":"유체화",
    "heal":"회복","ignite":"점화","cleanse":"정화","exhaust":"탈진","barrier":"방어막",
    "clarity":"총명","teleport":"순간이동",
}
# 표준 한글명 -> DDragon 키
KOR_TO_DDRAGON = {
    "점멸":"SummonerFlash",
    "표식":"SummonerSnowball",
    "유체화":"SummonerHaste",
    "회복":"SummonerHeal",
    "점화":"SummonerDot",
    "정화":"SummonerBoost",
    "탈진":"SummonerExhaust",
    "방어막":"SummonerBarrier",
    "총명":"SummonerMana",
    "순간이동":"SummonerTeleport",
}
DDRAGON_TO_KOR = {v:k for k,v in KOR_TO_DDRAGON.items()}

def standard_korean_spell(s: str) -> str:
    n = _norm(s)
    return SPELL_ALIASES.get(n, s)

def ddragon_spell_icon(name_or_kor: str) -> str:
    kor = standard_korean_spell(name_or_kor)
    key = KOR_TO_DDRAGON.get(kor)
    if not key:
        return ""
    return f"https://ddragon.leagueoflegends.com/cdn/{DD_VERSION}/img/spell/{key}.png"

def resolve_spell_icon(name: str) -> str:
    """1) spell_icons.csv → 2) 별칭 정규화 → 3) Data Dragon 폴백"""
    if not name:
        return ""
    raw = str(name).strip()
    for k in (raw, _norm(raw), standard_korean_spell(raw), _norm(standard_korean_spell(raw))):
        if k in spell_map:
            return spell_map[k]
    return ddragon_spell_icon(raw)

# 변경 ②: 스펠 조합을 '순서 무시'로 통합 집계
def canonical_spell_key(name: str) -> str:
    """조합 통합을 위한 키: DDragon 키가 있으면 그걸 쓰고, 없으면 소문자/공백제거명"""
    kor = standard_korean_spell(name)
    key = KOR_TO_DDRAGON.get(kor)
    return key if key else _norm(kor)

def pretty_spell_name_from_key(key: str) -> str:
    """DDragon 키면 한글로, 아니면 키 그대로"""
    return DDRAGON_TO_KOR.get(key, key)

def pick_spell_cols(df_):
    if {"spell1_name_fix","spell2_name_fix"}.issubset(df_.columns):
        return "spell1_name_fix", "spell2_name_fix"
    if {"spell1","spell2"}.issubset(df_.columns):
        return "spell1", "spell2"
    cands = [c for c in df_.columns if "spell" in c.lower()]
    return (cands[0], cands[1]) if len(cands) >= 2 else (None, None)

s1, s2 = pick_spell_cols(dsel)
if games and s1 and s2:
    tmp = dsel[[s1, s2, "win_clean"]].copy()
    tmp["k1"] = tmp[s1].astype(str).map(canonical_spell_key)
    tmp["k2"] = tmp[s2].astype(str).map(canonical_spell_key)
    # '순서 무시'를 위해 정렬된 키 쌍으로 대표키 생성
    tmp["pair_key"] = tmp.apply(lambda r: "|".join(sorted([r["k1"], r["k2"]])), axis=1)

    grp = (
        tmp.groupby("pair_key")
        .agg(games=("win_clean","count"), wins=("win_clean","sum"),
             k1=("k1","first"), k2=("k2","first"))
        .reset_index(drop=True)
    )
    grp["win_rate"] = (grp["wins"]/grp["games"]*100).round(2)

    # 표시용 이름/아이콘 (정렬된 키 기준으로 일관되게)
    grp[["disp_k1","disp_k2"]] = grp.apply(
        lambda r: pd.Series(sorted([r["k1"], r["k2"]])), axis=1
    )
    grp["spell1_name"] = grp["disp_k1"].map(pretty_spell_name_from_key)
    grp["spell2_name"] = grp["disp_k2"].map(pretty_spell_name_from_key)
    grp["spell1_icon"] = grp["spell1_name"].apply(resolve_spell_icon)
    grp["spell2_icon"] = grp["spell2_name"].apply(resolve_spell_icon)

    grp = grp.sort_values(["games","win_rate"], ascending=[False, False]).head(10)

    st.dataframe(
        grp[["spell1_icon","spell1_name","spell2_icon","spell2_name","games","wins","win_rate"]],
        use_container_width=True,
        column_config={
            "spell1_icon": st.column_config.ImageColumn("스펠1", width="small"),
            "spell2_icon": st.column_config.ImageColumn("스펠2", width="small"),
            "spell1_name": "스펠1 이름", "spell2_name": "스펠2 이름",
            "games":"게임수","wins":"승수","win_rate":"승률(%)"
        }
    )
else:
    st.info("스펠 컬럼을 찾지 못했습니다. (spell1_name_fix/spell2_name_fix 또는 spell1/spell2 필요)")

# ===== 룬 추천 =====
st.subheader("Recommended Rune Combos")
core_map = rune_maps.get("core", {})
sub_map  = rune_maps.get("sub", {})

def _rune_core_icon(name: str) -> str: return core_map.get(name, "")
def _rune_sub_icon(name: str)  -> str: return sub_map.get(name, "")

if games and {"rune_core","rune_sub"}.issubset(dsel.columns):
    ru = (
        dsel.groupby(["rune_core","rune_sub"])
        .agg(games=("win_clean","count"), wins=("win_clean","sum"))
        .reset_index()
    )
    ru["win_rate"] = (ru["wins"]/ru["games"]*100).round(2)
    ru = ru.sort_values(["games","win_rate"], ascending=[False,False]).head(10)
    ru["rune_core_icon"] = ru["rune_core"].apply(_rune_core_icon)
    ru["rune_sub_icon"]  = ru["rune_sub"].apply(_rune_sub_icon)

    st.dataframe(
        ru[["rune_core_icon","rune_core","rune_sub_icon","rune_sub","games","wins","win_rate"]].
        rename(columns={"rune_core":"핵심룬 이름","rune_sub":"보조트리 이름","games":"게임수","wins":"승수","win_rate":"승률(%)"}),
        use_container_width=True,
        column_config={
            "rune_core_icon": st.column_config.ImageColumn("핵심룬", width="small"),
            "rune_sub_icon":  st.column_config.ImageColumn("보조트리", width="small"),
        }
    )
else:
    st.info("룬 컬럼(rune_core, rune_sub)이 없습니다.")

# ===== 원본(선택 챔피언) =====
with st.expander("Raw rows (selected champion)"):
    st.dataframe(dsel, use_container_width=True)

