import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "../config_render";
const outputPath = `${outputDir}/Gryphon_math_sheet.xlsx`;
const previewPath = `${outputDir}/Gryphon_math_sheet_summary.png`;
const font = "Arial";

const tuning = {
  targetTotalRtp: 0.9402008490918528,
  baseLineRtp: 0.35119557999142814,
  baseCollectRtp: 0.20055065999993657,
  baseRtp: 0.5517462399913647,
  featureRtp: 0.3754137165425921,
  jackpotRtp: 0.013040892557895896,
  totalRtp: 0.9402008490918528,
  sessionsPerFeature: 50000,
};

const featureMeans = {
  Plain: 35.56842254698634,
  Splitter: 29.608688316054934,
  Grow: 34.7665338333386,
  Boost: 24.80206192508209,
  Multiplier: 28.424568514919448,
  Collect: 44.36037481526567,
  Expansion: 51.4483078772573,
  "Mega/Combo": 84.89215022880804,
};

const basePaytable = [
  ["H1", 0, 1.0, 3.0, 10.0],
  ["H2", 0, 0.6, 2.0, 5.0],
  ["H3", 0, 0.4, 1.6, 4.0],
  ["H4", 0, 0.4, 1.0, 3.0],
  ["H5", 0, 0.4, 1.0, 3.0],
  ["L1", 0, 0.2, 0.6, 2.0],
  ["L2", 0, 0.2, 0.6, 2.0],
  ["L3", 0, 0.2, 0.6, 2.0],
  ["L4", 0, 0.2, 0.6, 2.0],
  ["L5", 0, 0.2, 0.6, 2.0],
  ["L6", 0, 0.2, 0.6, 2.0],
  ["WD", 0, 0, 0, 0],
];

const baseCoinValues = [
  [0.2, 0.014581425011130],
  [0.3, 0.020756800175620],
  [0.5, 0.041403957230101],
  [0.8, 0.071979513760192],
  [0.9, 0.076847652889799],
  [1.0, 0.123067553102945],
  [1.2, 0.158980781060996],
  [1.5, 0.193468259037971],
  [2.0, 0.189430778346341],
  [2.5, 0.109483279384903],
];

const featureCoinValues = [
  [0.2, 0.035],
  [0.3, 0.040],
  [0.5, 0.060],
  [0.6, 0.075],
  [0.8, 0.090],
  [1.0, 0.130],
  [1.2, 0.165],
  [1.5, 0.160],
  [2.0, 0.120],
  [2.5, 0.075],
  [3.0, 0.050],
];

const growBoostValues = [
  [0.1, 0.08],
  [0.2, 0.12],
  [0.3, 0.16],
  [0.4, 0.18],
  [0.5, 0.16],
  [0.6, 0.13],
  [0.8, 0.10],
  [1.0, 0.07],
];

const comboBoostValues = [
  [0.1, 0.07],
  [0.2, 0.10],
  [0.3, 0.14],
  [0.4, 0.17],
  [0.5, 0.16],
  [0.6, 0.13],
  [0.8, 0.10],
  [1.0, 0.08],
  [1.5, 0.05],
];

const scatterConversion = [
  ["SC1 Splitter", 1 / 6],
  ["SC2 Grower", 1 / 6],
  ["SC3 Booster", 1 / 6],
  ["SC4 Multiplier", 1 / 6],
  ["SC5 Collector", 1 / 6],
  ["SC6 Expansion", 1 / 6],
];

const landingTables = [
  ["Expansion unlocked", "Empty", 0.650],
  ["Expansion unlocked", "Coin", 0.275],
  ["Expansion unlocked", "GO", 0.025],
  ["Expansion unlocked", "Mini token", 0.026],
  ["Expansion unlocked", "Minor token", 0.016],
  ["Expansion unlocked", "Major token", 0.006],
  ["Expansion unlocked", "Grand token", 0.002],
  ["Expansion locked", "Empty", 0.800],
  ["Expansion locked", "Coin", 0.145],
  ["Expansion locked", "GO", 0.010],
  ["Expansion locked", "Mini token", 0.026],
  ["Expansion locked", "Minor token", 0.014],
  ["Expansion locked", "Major token", 0.004],
  ["Expansion locked", "Grand token", 0.001],
  ["Standard 3x5", "Empty", 0.640],
  ["Standard 3x5", "Coin", 0.270],
  ["Standard 3x5", "Feature symbol", 0.035],
  ["Standard 3x5", "Mini token", 0.032],
  ["Standard 3x5", "Minor token", 0.016],
  ["Standard 3x5", "Major token", 0.005],
  ["Standard 3x5", "Grand token", 0.002],
  ["Mega unlocked", "Empty", 0.555],
  ["Mega unlocked", "Coin", 0.260],
  ["Mega unlocked", "GO", 0.025],
  ["Mega unlocked", "Grower", 0.030],
  ["Mega unlocked", "Collector", 0.030],
  ["Mega unlocked", "Splitter", 0.045],
  ["Mega unlocked", "Mini token", 0.032],
  ["Mega unlocked", "Minor token", 0.016],
  ["Mega unlocked", "Major token", 0.005],
  ["Mega unlocked", "Grand token", 0.002],
  ["Mega locked", "Empty", 0.715],
  ["Mega locked", "Coin", 0.170],
  ["Mega locked", "GO", 0.010],
  ["Mega locked", "Grower", 0.020],
  ["Mega locked", "Collector", 0.020],
  ["Mega locked", "Splitter", 0.025],
  ["Mega locked", "Mini token", 0.026],
  ["Mega locked", "Minor token", 0.010],
  ["Mega locked", "Major token", 0.003],
  ["Mega locked", "Grand token", 0.001],
];

const workbook = Workbook.create();

function setupSheet(sheet, widths = []) {
  sheet.showGridLines = false;
  widths.forEach((w, idx) => {
    sheet.getCell(0, idx).format.columnWidth = w;
  });
}

function title(sheet, text, range = "A2:H2") {
  sheet.getRange(range).merge();
  const cell = sheet.getRange(range.split(":")[0]);
  cell.values = [[text]];
  cell.format.font = { name: font, bold: true, size: 16, color: "#111827" };
  cell.format.fill = "#FFFFFF";
}

function table(sheet, anchor, headers, rows, name) {
  const start = sheet.getRange(anchor);
  start.write([headers, ...rows]);
  const rowCount = rows.length + 1;
  const colCount = headers.length;
  const range = start.resize(rowCount, colCount);
  const header = range.getRow(0);
  header.format.fill = "#1F4E78";
  header.format.font = { name: font, bold: true, color: "#FFFFFF", size: 10 };
  header.format.verticalAlignment = "center";
  range.format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
  range.format.autofitColumns();
  return range;
}

function sectionLabel(sheet, cell, text) {
  const r = sheet.getRange(cell);
  r.values = [[text]];
  r.format.fill = "#D9EAF7";
  r.format.font = { name: font, bold: true, color: "#17365D" };
}

const summary = workbook.worksheets.add("Summary");
setupSheet(summary, [22, 18, 22, 50]);
summary.tabColor = "#1F4E78";
title(summary, "Gryphon RTP tuning summary");
table(summary, "A4", ["Metric", "Value", "Note"], [
  ["Target total RTP", tuning.targetTotalRtp, "Paid-spin denominator"],
  ["Target base RTP", tuning.baseRtp, "Base line + Collect"],
  ["Target non-jackpot feature RTP", tuning.featureRtp, "Eight routed H&S lanes"],
  ["Target jackpot RTP", tuning.jackpotRtp, "Mini + Minor + Major + Grand"],
  ["Target allocation total", tuning.totalRtp, "Base + H&S + jackpots"],
  ["Overall feature odds", 0.01, "1 in 100"],
  ["Combo feature odds", 0.0004, "1 in 2,500"],
  ["Plain feature odds", 0.001248, "1 in 801.28"],
], "Summary");
summary.getRange("B5:B12").format.numberFormat = "0.0000%";
sectionLabel(summary, "A16", "Important odds note");
summary.getRange("A17:D18").merge();
summary.getRange("A17").values = [["Exactly one route runs per trigger: Mega is 0.04% of paid spins; the non-Mega allocation is 13% Plain and 87% split equally across the six single-Bag routes."]];
summary.getRange("A17").format.wrapText = true;
summary.getRange("A17").format.fill = "#FFF2CC";

const rtp = workbook.worksheets.add("RTP Build");
setupSheet(rtp, [18, 18, 18, 22, 20, 42]);
rtp.tabColor = "#5B9BD5";
title(rtp, "RTP build and feature odds");
table(rtp, "A4", ["Lane", "Requested odds", "Implemented odds", "Mean win", "RTP contribution", "Notes"], [
  ["Base line", "", 1, tuning.baseLineRtp, "", "Locked from 10M base validation"],
  ["Base Collect", "", 1, tuning.baseCollectRtp, "", "Locked from 10M base validation"],
  ["Plain H&S", 0.001248, 0.001248, featureMeans.Plain, "", "No-Bag route"],
  ["Splitter", 0.001392, 0.001392, featureMeans.Splitter, "", "SC1 route"],
  ["Grow", 0.001392, 0.001392, featureMeans.Grow, "", "SC2 route"],
  ["Boost", 0.001392, 0.001392, featureMeans.Boost, "", "SC3 route"],
  ["Multiplier", 0.001392, 0.001392, featureMeans.Multiplier, "", "SC4 route"],
  ["Collect", 0.001392, 0.001392, featureMeans.Collect, "", "SC5 route"],
  ["Expansion", 0.001392, 0.001392, featureMeans.Expansion, "", "SC6 route"],
  ["Mega/Combo", 0.0004, 0.0004, featureMeans["Mega/Combo"], "", "All-six route"],
  ["Mini jackpot", "", "", 0.00703751533863107, "", "Direct RTP target"],
  ["Minor jackpot", "", "", 0.00354982155325179, "", "Direct RTP target"],
  ["Major jackpot", "", "", 0.001852213908028074, "", "Direct RTP target"],
  ["Grand jackpot", "", "", 0.0006013417579849628, "", "Direct RTP target"],
  ["Total", "", "", "", "", "Formula total"],
], "RTPBuild");
rtp.getRange("E5").formulas = [["=D5"]];
rtp.getRange("E6").formulas = [["=D6"]];
rtp.getRange("E7").formulas = [["=C7*D7"]];
rtp.getRange("E7:E14").fillDown();
rtp.getRange("E15").formulas = [["=D15"]];
rtp.getRange("E15:E18").fillDown();
rtp.getRange("E19").formulas = [["=SUM(E5:E18)"]];
rtp.getRange("B7:C14").format.numberFormat = "0.0000%";
rtp.getRange("D5:E19").format.numberFormat = "0.0000%";
rtp.getRange("D7:D14").format.numberFormat = "0.0000";
sectionLabel(rtp, "A22", "Control values");
table(rtp, "A23", ["Control", "Value", "Meaning"], [
  ["Target total RTP", tuning.targetTotalRtp, "Requested"],
  ["Route multipliers", "Configured per route", "Never applied to jackpots"],
  ["Validation sessions per feature", tuning.sessionsPerFeature, "Feature-only sample size"],
], "Controls");
rtp.getRange("B24").format.numberFormat = "0.0000%";
rtp.getRange("B26").format.numberFormat = "#,##0";

const odds = workbook.worksheets.add("Odds");
setupSheet(odds, [28, 20, 20, 52]);
odds.tabColor = "#70AD47";
title(odds, "Feature route targets");
table(odds, "A4", ["Route", "Paid-spin probability", "Conditional mean", "RTP contribution"], [
  ["Plain", 0.001248, featureMeans.Plain, 0.044389391338638953],
  ["Splitter", 0.001392, featureMeans.Splitter, 0.041215294135948469],
  ["Grow", 0.001392, featureMeans.Grow, 0.048395015096007331],
  ["Boost", 0.001392, featureMeans.Boost, 0.034524470199714269],
  ["Multiplier", 0.001392, featureMeans.Multiplier, 0.039566999372767872],
  ["Collect", 0.001392, featureMeans.Collect, 0.061749641742849815],
  ["Expansion", 0.001392, featureMeans.Expansion, 0.071616044565142157],
  ["Mega Combo", 0.0004, featureMeans["Mega/Combo"], 0.033956860091523218],
], "Odds");
odds.getRange("B5:B12").format.numberFormat = "0.0000%";
odds.getRange("C5:C12").format.numberFormat = "0.0000";
odds.getRange("D5:D12").format.numberFormat = "0.0000%";

const base = workbook.worksheets.add("Base Arrays");
setupSheet(base, [16, 14, 14, 14, 14, 28, 16, 16, 16]);
base.tabColor = "#9DC3E6";
title(base, "Base game arrays and weights");
sectionLabel(base, "A4", "Paytable");
table(base, "A5", ["Symbol", "2OAK", "3OAK", "4OAK", "5OAK"], basePaytable, "BasePaytable");
sectionLabel(base, "G4", "Base coin values");
table(base, "G5", ["Value", "Probability"], baseCoinValues, "BaseCoinValues");
sectionLabel(base, "G18", "Coin drop count");
table(base, "G19", ["Count", "Probability"], [[2, 0.08], [3, 0.16], [4, 0.34], [5, 0.28], [6, 0.14]], "CoinDropCount");
sectionLabel(base, "A20", "Scatter conversion");
table(base, "A21", ["Converted symbol", "Probability"], scatterConversion, "ScatterConversion");
base.getRange("B21:B28").format.numberFormat = "0.0%";
base.getRange("H5:H14").format.numberFormat = "0.0%";
base.getRange("H20:H24").format.numberFormat = "0.0%";

const feature = workbook.worksheets.add("Feature Arrays");
setupSheet(feature, [26, 18, 18, 22, 20, 18, 18]);
feature.tabColor = "#A9D18E";
title(feature, "Feature arrays and weights");
sectionLabel(feature, "A4", "Feature coin values");
table(feature, "A5", ["Value", "Probability"], featureCoinValues, "FeatureCoinValues");
sectionLabel(feature, "D4", "Grow and boost values");
table(feature, "D5", ["Value", "Probability"], growBoostValues, "GrowBoostValues");
sectionLabel(feature, "A19", "Combo boost values");
table(feature, "A20", ["Value", "Probability"], comboBoostValues, "ComboBoostValues");
sectionLabel(feature, "D19", "Multiplier and splitter counts");
table(feature, "D20", ["Table", "Value", "Probability"], [
  ["Multiplier count", 4, 0.45],
  ["Multiplier count", 5, 0.35],
  ["Multiplier count", 6, 0.20],
  ["Multiplier value", 2, 0.70],
  ["Multiplier value", 3, 0.22],
  ["Multiplier value", 4, 0.08],
  ["Splitter coin count", 2, 0.70],
  ["Splitter coin count", 3, 0.30],
  ["Combo extra coin count", 1, 0.20],
  ["Combo extra coin count", 2, 0.40],
  ["Combo extra coin count", 3, 0.40],
], "FeatureCounts");
sectionLabel(feature, "A33", "Landing tables");
table(feature, "A34", ["Table", "Symbol", "Probability"], landingTables, "LandingTables");
feature.getRange("B5:B15").format.numberFormat = "0.0%";
feature.getRange("E5:E12").format.numberFormat = "0.0%";
feature.getRange("B20:B28").format.numberFormat = "0.0%";
feature.getRange("F20:F30").format.numberFormat = "0.0%";
feature.getRange("C34:C74").format.numberFormat = "0.0%";

const rules = workbook.worksheets.add("Game Rules");
setupSheet(rules, [26, 100]);
rules.tabColor = "#FFC000";
title(rules, "Game rules");
table(rules, "A4", ["Area", "Rule"], [
  ["Base", "Three configured reelsets use the normal Numba reel selection and 3x5 base-board path."],
  ["Feature odds", "Overall Hold-and-Spin odds are 1 in 100 paid spins."],
  ["Single features", "Each of the six single-Bag routes targets 0.1392% of paid spins."],
  ["Plain", "Plain Hold-and-Spin targets 0.1248% and starts without a Bag."],
  ["Combo", "Mega/Combo targets 0.04% and requires SC1-SC6 after base SC conversion."],
  ["Jackpots", "Mini, Minor, Major, and Grand require 3 matching tokens. Tokens are not affected by feature symbols."],
  ["Feature ending", "Feature sessions end when the grid is full or spins are exhausted."],
  ["Multiplier", "Multiplier cells affect coins only, not jackpot tokens."],
  ["Splitter", "Splitter cells hold 2 or 3 coin values and are not split again."],
  ["RTP tuning", "Each route has its own configured payout multiplier; progressive jackpots are never multiplied."],
], "Rules");
rules.getRange("B5:B14").format.wrapText = true;

const flow = workbook.worksheets.add("Game Flow");
setupSheet(flow, [20, 110]);
flow.tabColor = "#ED7D31";
title(flow, "Game flow");
table(flow, "A4", ["Flow", "Step"], [
  ["Base", "Select reelset, draw 5x3 window, evaluate lines, drop overlay coins, and resolve Collect awards."],
  ["Base trigger", "If generic SC appears, convert it to one of SC1-SC6 by weights."],
  ["Feature launch", "Select exactly one eligible route: Plain or one visible Bag; all six distinct Bags launch Mega."],
  ["Common feature start", "Start one shared Hold-and-Spin session with configured starting Coins and the selected route's Bag set."],
  ["Expansion", "6x5 grid with top 3 rows locked. GO unlocks next locked row and spins out."],
  ["Multiplier", "3x5 grid. Multiplier cells persist until a coin lands on them."],
  ["Grow", "Growers can independently grow 3-5 other normal coins on weighted checks."],
  ["Boost", "Boost spin increases normal coin values and converts booster to a normal coin."],
  ["Collect", "Collectors collect other normal coins, ignore jackpot tokens, and convert to normal coins."],
  ["Splitter", "Splitter cells hold 2 or 3 coin values. Minimum splitter rule is enforced."],
  ["Mega/Combo", "Start with SC1-SC6 on one 6x5-capable board; top three rows begin locked and Expansion can unlock them."],
], "Flow");
flow.getRange("B5:B15").format.wrapText = true;

for (const sheet of workbook.worksheets.items) {
  const used = sheet.getUsedRange();
  if (used) {
    used.format.verticalAlignment = "center";
    used.format.autofitColumns();
    used.format.autofitRows();
  }
}

workbook.recalculate();

await fs.mkdir(outputDir, { recursive: true });
const preview = await workbook.render({ sheetName: "Summary", autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

const inspect = await workbook.inspect({
  kind: "table",
  sheetId: "RTP Build",
  range: "A4:F19",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 8,
});
console.log(inspect.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(JSON.stringify({ outputPath, previewPath }));
