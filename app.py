# app.py — ARAM 대시보드 (필터 제거, 룬/스펠 역매핑 + 한글화)
import os, re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ARAM PS Dashboard", layout="wide")

# ===== 파일 경로 =====
PLAYERS_CSV = "aram_matches_superlight2.csv"   # 참가자 행 (슈퍼라이트2)
CHAMP_CSV   = "champion_icons.csv"             # champion, champion_icon (선택)
RUNE_CSV    = "rune_icons.csv"                 # (선택) 이름→아이콘 매핑
SPELL_CSV   = "spell_icons.csv"                # (선택) 스펠명→아이콘 매핑

DD_VERSION = "15.17.1"  # Data Dragon 폴백 버전

# ===== 유틸 =====
def _exists(path: str) -> bool:
    ok = os.path.exists(path)
    if not ok:
        st.warning(f"파일 없음: `{path}`")
    return ok

def _norm(x: str) -> str:
    return re.sub(r"\s+", "", str(x)).strip().lower()

def _looks_like_url(s: str) -> bool:
    return isinstance(s, str) and s.startswith(("http://", "https://"))

# ===== 로더 =====
@st.cache_data
def load_players(path: str) -> pd.DataFrame:
    if not _exists(path): st.stop()
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949")
    # 승패 정리
    if "win_clean" not in df.columns:
        df["win_clean"] = df.get("win","").astype(str).str.lower().isin(["true","1","t","yes"]).astype(int)
    # 문자열 정리
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].fillna("").astype(str).str.strip()
    return df

@st.cache_data
def load_champion_icons(path: str) -> dict:
    if not _exists(path): return {}
    df = pd.read_csv(path)
    name_col = next((c for c in ["champion","Champion","championName"] if c in df.columns), None)
    icon_col = next((c for c in ["champion_icon","icon","icon_url"] if c in df.columns), None)
    if not name_col or not icon_col: return {}
    df[name_col] = df[name_col].astype(str).str.strip()
    return dict(zip(df[name_col], df[icon_col]))

@st.cache_data
def load_rune_icons(path: str) -> dict:
    if not _exists(path): return {"core": {}, "sub": {}}
    df = pd.read_csv(path)
    core_map, sub_map = {}, {}
    if "rune_core" in df.columns:
        ic = "rune_core_icon" if "rune_core_icon" in df.columns else None
        if ic: core_map = dict(zip(df["rune_core"].astype(str), df[ic].astype(str)))
    if "rune_sub" in df.columns:
        ic = "rune_sub_icon" if "rune_sub_icon" in df.columns else None
        if ic: sub_map = dict(zip(df["rune_sub"].astype(str), df[ic].astype(str)))
    return {"core": core_map, "sub": sub_map}

@st.cache_data
def load_spell_icons(path: str) -> dict:
    if not _exists(path): return {}
    df = pd.read_csv(path)
    name_col = next((c for c in df.columns if _norm(c) in {"spell","spellname","name","스펠","스펠명"}), None)
    icon_col = next((c for c in df.columns if "icon" in c.lower()), None)
    m = {}
    if name_col and icon_col:
        for n, i in zip(df[name_col].astype(str), df[icon_col].astype(str)):
            m[_norm(n)] = i; m[str(n).strip()] = i
    elif df.shape[1] >= 2:
        for n, i in zip(df.iloc[:,0].astype(str), df.iloc[:,1].astype(str)):
            m[_norm(n)] = i; m[str(n).strip()] = i
    return m

# ===== 데이터 =====
df        = load_players(PLAYERS_CSV)
champ_map = load_champion_icons(CHAMP_CSV)
rune_maps = load_rune_icons(RUNE_CSV)
spell_map = load_spell_icons(SPELL_CSV)

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
    if cicon: st.image(cicon, width=64)
with ctitle:
    st.title(f"{selected}")

c1, c2, c3 = st.columns(3)
c1.metric("Games", f"{games}")
c2.metric("Win Rate", f"{winrate}%")
c3.metric("Pick Rate", f"{pickrate}%")

# ===== 아이템 추천 (슈퍼라이트2 내 아이콘/이름 사용) =====
st.subheader("Recommended Items")
item_name_cols = [c for c in dsel.columns if re.fullmatch(r"item[0-6]_name", c)]
item_icon_cols = [c for c in dsel.columns if re.fullmatch(r"item[0-6]_icon", c)]
if games and item_name_cols and item_icon_cols:
    stacks = []
    for i in range(7):
        ncol, icol = f"item{i}_name", f"item{i}_icon"
        if ncol in dsel.columns and icol in dsel.columns:
            part = dsel[[ncol, icol, "win_clean"]].rename(columns={ncol:"item", icol:"icon_url"})
            stacks.append(part)
    union = pd.concat(stacks, ignore_index=True)
    union = union[union["item"].astype(str).str.strip() != ""]
    top_items = (
        union.groupby(["item","icon_url"])
        .agg(total_picks=("item","count"), wins=("win_clean","sum"))
        .reset_index()
    )
    # 같은 이름의 여러 아이콘이 있으면 가장 많이 쓰인 아이콘 유지
    top_items = (
        top_items.sort_values(["item","total_picks"], ascending=[True, False])
        .drop_duplicates(subset=["item"])
    )
    top_items["win_rate"] = (top_items["wins"]/top_items["total_picks"]*100).round(2)
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
    st.info("아이템 이름/아이콘 컬럼이 없어 아이템 집계를 만들 수 없습니다. (item*_name, item*_icon 필요)")

# ===== 스펠 추천 (순서무시 통합 + URL/ID → 한글화) =====
st.subheader("Recommended Spell Combos")

# 표준 한글명 <-> DDragon 키
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
ID_TO_DDRAGON = {
    "4":"SummonerFlash", "32":"SummonerSnowball", "6":"SummonerHaste", "7":"SummonerHeal",
    "14":"SummonerDot", "1":"SummonerBoost", "3":"SummonerExhaust", "21":"SummonerBarrier",
    "13":"SummonerMana", "12":"SummonerTeleport"
}

def spell_icon_from_key(key: str) -> str:
    return f"https://ddragon.leagueoflegends.com/cdn/{DD_VERSION}/img/spell/{key}.png" if key else ""

def pick_spell_source(df_):
    # 1) name_fix 2) spell1/2가 URL이면 -> id 3) spell1/2 이름
    if {"spell1_name_fix","spell2_name_fix"}.issubset(df_.columns):
        return ("name", "spell1_name_fix", "spell2_name_fix")
    if {"spell1","spell2"}.issubset(df_.columns):
        if len(df_) > 0 and (_looks_like_url(df_["spell1"].astype(str).iloc[0]) or _looks_like_url(df_["spell2"].astype(str).iloc[0])):
            if {"spell1Id","spell2Id"}.issubset(df_.columns):
                return ("id", "spell1Id", "spell2Id")
        return ("name", "spell1", "spell2")
    return (None, None, None)

mode, col1, col2 = pick_spell_source(dsel)
if games and mode:
    tmp = dsel[[col1, col2, "win_clean"]].copy()
    if mode == "id":
        tmp["k1"] = tmp[col1].astype(str).map(ID_TO_DDRAGON).fillna("")
        tmp["k2"] = tmp[col2].astype(str).map(ID_TO_DDRAGON).fillna("")
        tmp["n1"] = tmp["k1"].map(DDRAGON_TO_KOR).fillna(tmp["k1"])
        tmp["n2"] = tmp["k2"].map(DDRAGON_TO_KOR).fillna(tmp["k2"])
    else:
        def to_key(name: str) -> str:
            n = str(name).strip()
            return KOR_TO_DDRAGON.get(n, "")
        tmp["k1"] = tmp[col1].apply(to_key)
        tmp["k2"] = tmp[col2].apply(to_key)
        tmp["n1"] = tmp[col1].astype(str)
        tmp["n2"] = tmp[col2].astype(str)

    # 순서무시 대표키
    tmp["pair_key"] = tmp.apply(lambda r: "|".join(sorted([r["k1"], r["k2"]])), axis=1)
    grp = (
        tmp.groupby("pair_key")
        .agg(games=("win_clean","count"), wins=("win_clean","sum"),
             k1=("k1","first"), k2=("k2","first"),
             n1=("n1","first"), n2=("n2","first"))
        .reset_index(drop=True)
    )
    grp["win_rate"] = (grp["wins"]/grp["games"]*100).round(2)

    # 아이콘/이름 (키 기준)
    grp[["disp_k1","disp_k2"]] = grp.apply(lambda r: pd.Series(sorted([r["k1"], r["k2"]])), axis=1)
    grp["spell1_icon"] = grp["disp_k1"].apply(spell_icon_from_key)
    grp["spell2_icon"] = grp["disp_k2"].apply(spell_icon_from_key)
    grp["spell1_name"] = grp["disp_k1"].map(DDRAGON_TO_KOR).fillna(grp["n1"])
    grp["spell2_name"] = grp["disp_k2"].map(DDRAGON_TO_KOR).fillna(grp["n2"])

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
    st.info("스펠 컬럼을 찾지 못했습니다. (spell1Id or spell1_name_fix / spell1 필요)")

# ===== 룬 추천 (URL → 이름 역매핑: 트리/키스톤, 한글화) =====
st.subheader("Recommended Rune Combos")

# 트리 영→한
STYLE_EN2KO = {
    "Precision":"정밀", "Domination":"지배", "Resolve":"결의", "Sorcery":"마법", "Inspiration":"영감"
}
# 키스톤 영→한
KEYSTONE_EN2KO = {
    # Precision
    "PressTheAttack":"집중 공격", "LethalTempo":"치명적 속도", "FleetFootwork":"기민한 발놀림", "Conqueror":"정복자",
    # Domination
    "Electrocute":"감전", "DarkHarvest":"어둠의 수확", "HailOfBlades":"칼날비",
    # Sorcery
    "SummonAery":"콩콩이 소환", "ArcaneComet":"신비로운 유성", "PhaseRush":"난입",
    # Resolve
    "GraspOfTheUndying":"착취의 손아귀", "Aftershock":"여진", "Guardian":"수호자",
    # Inspiration
    "GlacialAugment":"빙결 강화", "UnsealedSpellbook":"봉인 풀린 주문서", "FirstStrike":"선제공격",
}

def infer_style_key(val: str) -> str:
    if not isinstance(val, str): return ""
    # .../Styles/7201_Precision.png  또는 .../Styles/Precision/...
    m = re.search(r'(?:_|/)(Precision|Domination|Resolve|Sorcery|Inspiration)(?:\.png|/)', val)
    return m.group(1) if m else ""

def infer_keystone_key(val: str) -> str:
    if not isinstance(val, str): return ""
    # .../Styles/Resolve/Aftershock/Aftershock.png  또는 마지막 파일명
    m = re.search(r'/Styles/(?:Precision|Domination|Resolve|Sorcery|Inspiration)/([A-Za-z0-9]+)/', val)
    key = m.group(1) if m else ""
    if not key:
        m2 = re.search(r'/([A-Za-z0-9]+)\.png$', val)
        key = m2.group(1) if m2 else ""
    return key

if games and {"rune_core","rune_sub"}.issubset(dsel.columns):
    ru = (
        dsel.groupby(["rune_core","rune_sub"])
        .agg(games=("win_clean","count"), wins=("win_clean","sum"))
        .reset_index()
    )
    ru["win_rate"] = (ru["wins"]/ru["games"]*100).round(2)

    # 행 단위 URL 판별
    core_is_url = ru["rune_core"].astype(str).str.startswith(("http://","https://"))
    sub_is_url  = ru["rune_sub"].astype(str).str.startswith(("http://","https://"))

    # 외부 매핑 (있을 때만 사용)
    core_map = rune_maps.get("core", {})
    sub_map  = rune_maps.get("sub",  {})

    # 아이콘
    ru["rune_core_icon"] = ru["rune_core"].where(core_is_url, ru["rune_core"].map(core_map).fillna(""))
    ru["rune_sub_icon"]  = ru["rune_sub"].where(sub_is_url,  ru["rune_sub"].map(sub_map).fillna(""))

    # 이름(역매핑 → 한글화)
    # core: keystone, sub: style
    ru["rune_core_name"] = ru["rune_core"].where(~core_is_url, ru["rune_core"].apply(lambda v: KEYSTONE_EN2KO.get(infer_keystone_key(v), infer_keystone_key(v))))
    ru["rune_sub_name"]  = ru["rune_sub"].where(~sub_is_url,  ru["rune_sub"].apply(lambda v: STYLE_EN2KO.get(infer_style_key(v), infer_style_key(v))))

    # URL이 아니라 영어 키/이름이 이미 들어온 경우도 한글화 시도
    ru.loc[~core_is_url, "rune_core_name"] = ru.loc[~core_is_url, "rune_core"].map(KEYSTONE_EN2KO).fillna(ru.loc[~core_is_url, "rune_core"])
    ru.loc[~sub_is_url,  "rune_sub_name"]  = ru.loc[~sub_is_url,  "rune_sub"].map(STYLE_EN2KO).fillna(ru.loc[~sub_is_url,  "rune_sub"])

    ru = ru.sort_values(["games","win_rate"], ascending=[False,False]).head(10)
    st.dataframe(
        ru[["rune_core_icon","rune_core_name","rune_sub_icon","rune_sub_name","games","wins","win_rate"]],
        use_container_width=True,
        column_config={
            "rune_core_icon": st.column_config.ImageColumn("핵심룬", width="small"),
            "rune_sub_icon":  st.column_config.ImageColumn("보조트리", width="small"),
            "rune_core_name":"핵심룬 이름","rune_sub_name":"보조트리 이름",
            "games":"게임수","wins":"승수","win_rate":"승률(%)"
        }
    )
else:
    st.info("룬 컬럼(rune_core, rune_sub)이 없습니다.")

# ===== 원본(선택 챔피언) =====
with st.expander("Raw rows (selected champion)"):
    st.dataframe(dsel, use_container_width=True)
