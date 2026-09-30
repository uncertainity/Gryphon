import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "../config_render";
const outputPath = `${outputDir}/Gryphon_math_sheet.xlsx`;
const previewPath = `${outputDir}/Gryphon_math_sheet_summary.png`;
const font = "Arial";

const tuning = {
  targetTotalRtp: 0.94,
  baseLineRtp: 0.35119557999142814,
  baseCollectRtp: 0.20055065999993657,
  baseRtp: 0.5517462399913647,
  targetFeatureRtp: 0.3882537600086353,
  featureRtp: 0.388454609100488,
  totalRtp: 0.9402008490918528,
  featurePayoutMultiplier: 1.086911,
  sessionsPerFeature: 5000,
};

const featureMeans = {
  Splitter: 30.63721685962,
  Grow: 35.974232466480004,
  Boost: 25.66362081472,
  Multiplier: 29.41196382754,
  Collect: 45.901338441,
  Expansion: 53.2354877957,
  "Mega/Combo": 87.84108193098,
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
  [0.2, 0.030],
  [0.3, 0.040],
  [0.5, 0.070],
  [0.8, 0.100],
  [0.9, 0.100],
  [1.0, 0.150],
  [1.2, 0.170],
  [1.5, 0.170],
  [2.0, 0.120],
  [2.5, 0.050],
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
  ["SC1 Splitter", 0.145],
  ["SC2 Grower", 0.145],
  ["SC3 Booster", 0.145],
  ["SC4 Multiplier", 0.145],
  ["SC5 Collector", 0.145],
  ["SC6 Expansion", 0.145],
  ["SC7 Combo marker", 0.130],
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
  ["Target total RTP", tuning.targetTotalRtp, "User target"],
  ["Locked base RTP", tuning.baseRtp, "Base line + Collect. Base settings unchanged."],
  ["Target feature RTP", tuning.targetFeatureRtp, "Target total less locked base"],
  ["Tuned feature RTP", tuning.featureRtp, "5,000 sessions per feature lane"],
  ["Tuned total RTP", tuning.totalRtp, "Locked base plus tuned feature layer"],
  ["Feature payout multiplier", tuning.featurePayoutMultiplier, "Applied to final feature win only"],
  ["Overall feature odds", 0.01, "1 in 100"],
  ["Combo feature odds", 0.0004, "1 in 2,500"],
], "Summary");
summary.getRange("B5:B9").format.numberFormat = "0.0000%";
summary.getRange("B10").format.numberFormat = "0.000000";
summary.getRange("B11:B12").format.numberFormat = "0.0000%";
sectionLabel(summary, "A16", "Important odds note");
summary.getRange("A17:D18").merge();
summary.getRange("A17").values = [["Six single features at exactly 1 in 600 already sum to 1 in 100. Adding combo at 1 in 2,500 would make total odds 1 in 96.15. This tuning preserves total 1 in 100 and combo 1 in 2,500 by using single-feature lanes at 1 in 625 each."]];
summary.getRange("A17").format.wrapText = true;
summary.getRange("A17").format.fill = "#FFF2CC";

const rtp = workbook.worksheets.add("RTP Build");
setupSheet(rtp, [18, 18, 18, 22, 20, 42]);
rtp.tabColor = "#5B9BD5";
title(rtp, "RTP build and feature odds");
table(rtp, "A4", ["Lane", "Requested odds", "Implemented odds", "Mean win", "RTP contribution", "Notes"], [
  ["Base line", "", 1, tuning.baseLineRtp, "", "Locked from 10M base validation"],
  ["Base Collect", "", 1, tuning.baseCollectRtp, "", "Locked from 10M base validation"],
  ["Splitter", 1 / 600, 1 / 625, featureMeans.Splitter, "", "Single feature lane"],
  ["Grow", 1 / 600, 1 / 625, featureMeans.Grow, "", "Single feature lane"],
  ["Boost", 1 / 600, 1 / 625, featureMeans.Boost, "", "Single feature lane"],
  ["Multiplier", 1 / 600, 1 / 625, featureMeans.Multiplier, "", "Single feature lane"],
  ["Collect", 1 / 600, 1 / 625, featureMeans.Collect, "", "Single feature lane"],
  ["Expansion", 1 / 600, 1 / 625, featureMeans.Expansion, "", "Single feature lane"],
  ["Mega/Combo", 1 / 2500, 1 / 2500, featureMeans["Mega/Combo"], "", "Combo lane"],
  ["Total", "", "", "", "", "Formula total"],
], "RTPBuild");
rtp.getRange("E5").formulas = [["=D5"]];
rtp.getRange("E6").formulas = [["=D6"]];
rtp.getRange("E7").formulas = [["=C7*D7"]];
rtp.getRange("E7:E13").fillDown();
rtp.getRange("E14").formulas = [["=SUM(E5:E13)"]];
rtp.getRange("B7:C13").format.numberFormat = "0.0000%";
rtp.getRange("D5:E14").format.numberFormat = "0.0000%";
rtp.getRange("D7:D13").format.numberFormat = "0.0000";
sectionLabel(rtp, "A17", "Control values");
table(rtp, "A18", ["Control", "Value", "Meaning"], [
  ["Target total RTP", tuning.targetTotalRtp, "Requested"],
  ["Feature payout multiplier", tuning.featurePayoutMultiplier, "Applied to all feature final wins"],
  ["Validation sessions per feature", tuning.sessionsPerFeature, "Feature-only sample size"],
], "Controls");
rtp.getRange("B19").format.numberFormat = "0.0000%";
rtp.getRange("B20").format.numberFormat = "0.000000";
rtp.getRange("B21").format.numberFormat = "#,##0";

const odds = workbook.worksheets.add("Odds");
setupSheet(odds, [28, 20, 20, 52]);
odds.tabColor = "#70AD47";
title(odds, "Feature odds reconciliation");
table(odds, "A4", ["Scenario", "Single lane odds", "Combo odds", "Total feature odds", "Result"], [
  ["Requested singles plus combo", 1 / 600, 1 / 2500, 6 / 600 + 1 / 2500, "Total becomes 1 in 96.15"],
  ["Implemented tuning", 1 / 625, 1 / 2500, 6 / 625 + 1 / 2500, "Total remains 1 in 100"],
], "Odds");
odds.getRange("B5:D6").format.numberFormat = "0.0000%";

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
  ["Base", "Base game settings, reelsets, paytable, Collect mechanic, and coin-drop settings remain unchanged."],
  ["Feature odds", "Overall feature odds are kept at 1 in 100 in the implemented tuning."],
  ["Single features", "Six single-feature lanes are tuned at 1 in 625 each to preserve the total feature odds with combo included."],
  ["Requested odds note", "Exact 1 in 600 for all six singles plus 1 in 2,500 combo cannot also equal overall 1 in 100."],
  ["Combo", "Mega/Combo triggers only when SC1-SC6 are all present after base SC conversion."],
  ["Jackpots", "Mini, Minor, Major, and Grand require 3 matching tokens. Tokens are not affected by feature symbols."],
  ["Feature ending", "Feature sessions end when the grid is full or spins are exhausted."],
  ["Multiplier", "Multiplier cells affect coins only, not jackpot tokens."],
  ["Splitter", "Splitter cells hold 2 or 3 coin values and are not split again."],
  ["RTP tuning", "Feature payout multiplier of 1.086911x is applied to final feature wins only."],
], "Rules");
rules.getRange("B5:B14").format.wrapText = true;

const flow = workbook.worksheets.add("Game Flow");
setupSheet(flow, [20, 110]);
flow.tabColor = "#ED7D31";
title(flow, "Game flow");
table(flow, "A4", ["Flow", "Step"], [
  ["Base", "Select reelset, draw 5x3 window, evaluate lines, drop overlay coins, and resolve Collect awards."],
  ["Base trigger", "If generic SC appears, convert it to SC1-SC7 by weights for feature routing."],
  ["Feature launch", "Use the odds table for feature selection. Combo requires SC1-SC6 and uses 1 in 2,500 odds."],
  ["Common feature start", "Trigger symbol enters active area. Add three normal coins for single features or 1/2/3 extra coins for combo."],
  ["Expansion", "6x5 grid with top 3 rows locked. GO unlocks next locked row and spins out."],
  ["Multiplier", "3x5 grid. Multiplier cells persist until a coin lands on them."],
  ["Grow", "Growers can independently grow 3-5 other normal coins on weighted checks."],
  ["Boost", "Boost spin increases normal coin values and converts booster to a normal coin."],
  ["Collect", "Collectors collect other normal coins, ignore jackpot tokens, and convert to normal coins."],
  ["Splitter", "Splitter cells hold 2 or 3 coin values. Minimum splitter rule is enforced."],
  ["Mega/Combo", "SC1-SC6 keep exact base positions in the bottom three rows, perform guaranteed opening actions, then continue as a combined 6x5 feature."],
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
  range: "A4:F14",
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
