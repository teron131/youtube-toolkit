// Drive viewer state, rendering, and browser interactions against the read-only JSON API.

const THEME_STORAGE_KEY = "lancedb-viewer-theme";
const systemTheme = window.matchMedia("(prefers-color-scheme: dark)");

const state = {
	overview: null,
	table: null,
	detail: null,
	page: null,
	offset: 0,
	limit: 25,
	sortColumn: null,
	sortDirection: "asc",
	searchValue: "",
	searchTimer: null,
	rowsRequest: 0,
	filterColumn: null,
	filterOperator: null,
	filterValue: null,
	filterVisible: false,
	tab: "data",
	liveTimer: null,
	dialogJson: "",
	selectedRowNumber: null,
	inspectorTrigger: null,
};

const elements = {
	databasePath: document.getElementById("databasePath"),
	refreshButton: document.getElementById("refreshButton"),
	themeToggle: document.getElementById("themeToggle"),
	lightThemeIcon: document.getElementById("lightThemeIcon"),
	darkThemeIcon: document.getElementById("darkThemeIcon"),
	liveRefresh: document.getElementById("liveRefresh"),
	tableSearch: document.getElementById("tableSearch"),
	tableList: document.getElementById("tableList"),
	tableCount: document.getElementById("tableCount"),
	rowCount: document.getElementById("rowCount"),
	indexCount: document.getElementById("indexCount"),
	databaseSize: document.getElementById("databaseSize"),
	tableTitle: document.getElementById("tableTitle"),
	tableMeta: document.getElementById("tableMeta"),
	dataSearchControls: document.getElementById("dataSearchControls"),
	rowSearch: document.getElementById("rowSearch"),
	clearSearch: document.getElementById("clearSearch"),
	filterToggle: document.getElementById("filterToggle"),
	filterToggleText: document.getElementById("filterToggleText"),
	filterForm: document.getElementById("filterForm"),
	closeFilter: document.getElementById("closeFilter"),
	filterColumn: document.getElementById("filterColumn"),
	filterOperator: document.getElementById("filterOperator"),
	filterValue: document.getElementById("filterValue"),
	clearFilter: document.getElementById("clearFilter"),
	errorBanner: document.getElementById("errorBanner"),
	dataPanel: document.getElementById("dataPanel"),
	structurePanel: document.getElementById("structurePanel"),
	tableScroll: document.getElementById("tableScroll"),
	pageSize: document.getElementById("pageSize"),
	previousPage: document.getElementById("previousPage"),
	nextPage: document.getElementById("nextPage"),
	pagerInfo: document.getElementById("pagerInfo"),
	rowInspector: document.getElementById("rowInspector"),
	dialogTitle: document.getElementById("dialogTitle"),
	rawJson: document.getElementById("rawJson"),
	copyRow: document.getElementById("copyRow"),
	closeDialog: document.getElementById("closeDialog"),
};

function storedTheme() {
	const theme = window.localStorage.getItem(THEME_STORAGE_KEY);
	return theme === "dark" || theme === "light" ? theme : null;
}

function renderTheme(theme) {
	document.documentElement.dataset.theme = theme;
	const nextTheme = theme === "dark" ? "light" : "dark";
	const label = `Use ${nextTheme} theme`;
	elements.themeToggle.setAttribute("aria-label", label);
	elements.themeToggle.title = label;
	elements.lightThemeIcon.hidden = nextTheme !== "light";
	elements.darkThemeIcon.hidden = nextTheme !== "dark";
}

function useSystemTheme() {
	if (!storedTheme()) renderTheme(systemTheme.matches ? "dark" : "light");
}

renderTheme(storedTheme() || (systemTheme.matches ? "dark" : "light"));

function node(tag, className, text) {
	const element = document.createElement(tag);
	if (className) element.className = className;
	if (text !== undefined) element.textContent = text;
	return element;
}

function metricItem(value, label) {
	const item = node("span", "summary-item");
	item.append(
		node("span", "summary-value", value),
		node("span", "summary-name", label),
	);
	return item;
}

function replaceChildren(parent, children) {
	parent.replaceChildren(...children);
}

function formatInteger(value) {
	return new Intl.NumberFormat("en-US").format(value || 0);
}

function formatBytes(value) {
	if (!value) return "0 B";
	const units = ["B", "KB", "MB", "GB", "TB"];
	const exponent = Math.min(
		Math.floor(Math.log(value) / Math.log(1024)),
		units.length - 1,
	);
	const amount = value / 1024 ** exponent;
	return `${amount >= 10 || exponent === 0 ? amount.toFixed(0) : amount.toFixed(1)} ${units[exponent]}`;
}

function compactDatabasePath(value) {
	const parts = value.split(/[\\/]/).filter(Boolean);
	return parts.slice(-2).join("/") || value;
}

async function requestJson(path) {
	const response = await fetch(path, { cache: "no-store" });
	let payload;
	try {
		payload = await response.json();
	} catch {
		throw new Error(`Viewer returned HTTP ${response.status}`);
	}
	if (!response.ok)
		throw new Error(payload.error || `Viewer returned HTTP ${response.status}`);
	return payload;
}

function showError(error) {
	elements.errorBanner.textContent =
		error instanceof Error ? error.message : String(error);
	elements.errorBanner.hidden = false;
}

function clearError() {
	elements.errorBanner.hidden = true;
	elements.errorBanner.textContent = "";
}

async function loadOverview({ selectInitial = false } = {}) {
	const overview = await requestJson("/api/overview");
	state.overview = overview;
	renderOverview();
	if ((selectInitial || !state.table) && overview.tables.length > 0) {
		const initialTable =
			overview.tables.find((table) => table.name === "videos") ||
			overview.tables[0];
		await selectTable(initialTable.name);
	}
}

function renderOverview() {
	const overview = state.overview;
	elements.databasePath.textContent = compactDatabasePath(overview.path);
	elements.databasePath.title = overview.path;
	elements.tableCount.textContent = formatInteger(overview.table_count);
	elements.rowCount.textContent = formatInteger(overview.row_count);
	elements.indexCount.textContent = formatInteger(overview.index_count);
	elements.databaseSize.textContent = formatBytes(overview.bytes);
	renderTableList();
}

function renderTableList() {
	const term = elements.tableSearch.value.trim().toLocaleLowerCase();
	const links = state.overview.tables
		.filter((table) => table.name.toLocaleLowerCase().includes(term))
		.map((table) => {
			const button = node(
				"button",
				`table-link${table.name === state.table ? " active" : ""}`,
			);
			button.type = "button";
			button.dataset.table = table.name;
			if (table.name === state.table)
				button.setAttribute("aria-current", "page");
			button.append(node("span", "table-link-name", table.name));
			button.append(
				node("span", "table-link-count", formatInteger(table.rows)),
			);
			button.addEventListener("click", () => selectTable(table.name));
			return button;
		});
	replaceChildren(
		elements.tableList,
		links.length ? links : [node("div", "empty-state", "No matching tables")],
	);
}

async function selectTable(tableName) {
	closeRowInspector();
	state.table = tableName;
	state.offset = 0;
	state.sortColumn = null;
	state.sortDirection = "asc";
	state.searchValue = "";
	state.filterColumn = null;
	state.filterOperator = null;
	state.filterValue = null;
	state.filterVisible = false;
	elements.rowSearch.value = "";
	renderTableList();
	elements.tableTitle.textContent = tableName;
	elements.tableMeta.textContent = "Loading table metadata";
	elements.tableScroll.replaceChildren(
		node("div", "loading-state", "Loading live rows"),
	);
	try {
		clearError();
		const encoded = encodeURIComponent(tableName);
		const [detail, page] = await Promise.all([
			requestJson(`/api/tables/${encoded}`),
			requestJson(`/api/tables/${encoded}/rows?limit=${state.limit}&offset=0`),
		]);
		state.detail = detail;
		state.page = page;
		renderCurrentTable();
	} catch (error) {
		showError(error);
	}
}

async function loadRows() {
	if (!state.table) return;
	const requestId = ++state.rowsRequest;
	closeRowInspector();
	elements.tableScroll.setAttribute("aria-busy", "true");
	try {
		clearError();
		const params = new URLSearchParams({
			offset: String(state.offset),
			limit: String(state.limit),
			direction: state.sortDirection,
		});
		if (state.sortColumn) params.set("sort", state.sortColumn);
		if (state.searchValue) params.set("search", state.searchValue);
		if (state.filterColumn && state.filterOperator) {
			params.set("filter_column", state.filterColumn);
			params.set("filter_operator", state.filterOperator);
			if (state.filterValue !== null)
				params.set("filter_value", state.filterValue);
		}
		const page = await requestJson(
			`/api/tables/${encodeURIComponent(state.table)}/rows?${params}`,
		);
		if (requestId !== state.rowsRequest) return;
		state.page = page;
		syncSearchControls();
		renderDataTable();
		renderPager();
	} catch (error) {
		if (requestId === state.rowsRequest) showError(error);
	} finally {
		if (requestId === state.rowsRequest)
			elements.tableScroll.removeAttribute("aria-busy");
	}
}

function renderCurrentTable() {
	const overviewTable = state.overview.tables.find(
		(table) => table.name === state.table,
	);
	elements.tableTitle.textContent = state.table;
	replaceChildren(elements.tableMeta, [
		metricItem(formatInteger(overviewTable?.rows || 0), "rows"),
		metricItem(formatInteger(state.detail.schema.length), "columns"),
	]);
	populateFilterControls();
	syncSearchControls();
	renderDataTable();
	renderPager();
	renderStructure();
	activateTab(state.tab);
}

function populateFilterControls() {
	const options = [new Option("Choose a column", "")];
	for (const field of state.detail.schema.filter((item) => item.filterable)) {
		options.push(new Option(`${field.name}  (${field.type})`, field.name));
	}
	replaceChildren(elements.filterColumn, options);
	elements.filterColumn.value = state.filterColumn || "";
	renderFilterOperators();
	elements.filterValue.value = state.filterValue || "";
}

function renderFilterOperators() {
	const field = state.detail?.schema.find(
		(item) => item.name === elements.filterColumn.value,
	);
	const operators = field?.text
		? [
				["contains", "contains"],
				["equals", "equals"],
				["is_null", "is null"],
				["is_not_null", "is not null"],
			]
		: [
				["equals", "equals"],
				["is_null", "is null"],
				["is_not_null", "is not null"],
			];
	replaceChildren(
		elements.filterOperator,
		operators.map(([value, label]) => new Option(label, value)),
	);
	if (
		state.filterOperator &&
		operators.some(([value]) => value === state.filterOperator)
	) {
		elements.filterOperator.value = state.filterOperator;
	}
	const needsValue = !["is_null", "is_not_null"].includes(
		elements.filterOperator.value,
	);
	elements.filterValue.disabled = !needsValue;
}

function syncSearchControls() {
	const searchableColumns =
		state.detail?.schema.filter((field) => field.filterable).length || 0;
	const hasSemanticSearch = Boolean(state.detail?.semantic_search);
	elements.rowSearch.disabled = searchableColumns === 0;
	elements.rowSearch.placeholder = hasSemanticSearch
		? "Search values or meaning"
		: searchableColumns
			? `Search ${searchableColumns} ${searchableColumns === 1 ? "column" : "columns"}`
			: "No scalar columns to search";
	const semanticResult = state.page?.semantic_search;
	elements.rowSearch.title = hasSemanticSearch
		? semanticResult?.status === "unavailable"
			? `Literal results only; semantic search unavailable: ${semanticResult.detail}`
			: "Literal values plus semantic transcript matches"
		: searchableColumns
			? "Literal text and exact scalar value search"
			: "This table has no searchable scalar columns";
	elements.clearSearch.hidden = !state.searchValue;
	const hasFilter = Boolean(state.filterColumn && state.filterOperator);
	elements.filterToggle.classList.toggle("has-filter", hasFilter);
	elements.filterToggle.setAttribute(
		"aria-expanded",
		String(state.filterVisible),
	);
	elements.filterToggleText.textContent = hasFilter ? "1 filter" : "Filter";
	elements.filterToggle.title = hasFilter
		? `${state.filterColumn} ${state.filterOperator}${state.filterValue === null ? "" : ` ${state.filterValue}`}`
		: "Filter rows by one column rule";
	elements.clearFilter.disabled = !hasFilter;
	elements.filterForm.hidden = state.tab !== "data" || !state.filterVisible;
}

/** Close the anchored filter editor and optionally restore focus to its trigger. */
function closeFilter({ restoreFocus = false } = {}) {
	state.filterVisible = false;
	syncSearchControls();
	if (restoreFocus) elements.filterToggle.focus();
}

function renderDataTable() {
	if (!state.detail || !state.page) return;
	if (state.page.rows.length === 0) {
		const emptyMessage = state.searchValue
			? `No rows match “${truncate(state.searchValue, 48)}”`
			: state.page.total === 0
				? "This table has no matching rows"
				: "This page has no rows";
		elements.tableScroll.replaceChildren(
			node("div", "empty-state", emptyMessage),
		);
		return;
	}

	const table = node("table");
	table.setAttribute("aria-label", `${state.table} rows`);
	const thead = node("thead");
	const headerRow = node("tr");
	headerRow.append(node("th", "", "ROW"));
	for (const field of state.detail.schema) {
		const th = node("th");
		if (field.sortable) {
			const button = node("button", "sort-button");
			button.type = "button";
			button.append(node("span", "", field.name));
			const indicator = node(
				"span",
				"sort-indicator",
				state.sortColumn === field.name
					? state.sortDirection === "asc"
						? "ASC"
						: "DESC"
					: "",
			);
			button.append(indicator);
			button.addEventListener("click", () => changeSort(field.name));
			th.append(button);
			th.setAttribute(
				"aria-sort",
				state.sortColumn === field.name
					? state.sortDirection === "asc"
						? "ascending"
						: "descending"
					: "none",
			);
		} else {
			th.textContent = field.name;
		}
		headerRow.append(th);
	}
	thead.append(headerRow);
	table.append(thead);

	const tbody = node("tbody");
	state.page.rows.forEach((row, index) => {
		const tr = node("tr");
		tr.dataset.rowNumber = String(state.offset + index + 1);
		const actionCell = node("td");
		const rowButton = node(
			"button",
			"row-open",
			String(state.offset + index + 1),
		);
		rowButton.type = "button";
		rowButton.title = `Open raw row ${state.offset + index + 1}`;
		rowButton.setAttribute(
			"aria-label",
			`Open raw row ${state.offset + index + 1}`,
		);
		rowButton.setAttribute("aria-controls", "rowInspector");
		rowButton.setAttribute("aria-expanded", "false");
		rowButton.addEventListener("click", () =>
			openRow(row, state.offset + index + 1, rowButton),
		);
		actionCell.append(rowButton);
		tr.append(actionCell);
		for (const field of state.detail.schema) {
			const value = row[field.name];
			const td = node("td", cellClass(value));
			renderCellValue(td, value);
			tr.append(td);
		}
		tbody.append(tr);
	});
	table.append(tbody);
	elements.tableScroll.replaceChildren(table);
}

function cellClass(value) {
	if (value === null || value === undefined) return "cell-null";
	if (Array.isArray(value) || typeof value === "object") return "cell-nested";
	if (typeof value === "string") return "cell-text";
	return "";
}

function displayValue(value) {
	if (value === null || value === undefined) return "null";
	if (Array.isArray(value)) {
		const preview = value
			.slice(0, 3)
			.map((item) => compactValue(item))
			.join(", ");
		return `[${preview}${value.length > 3 ? ", ..." : ""}]  ${value.length} items`;
	}
	if (typeof value === "object") return truncate(JSON.stringify(value), 140);
	if (typeof value === "string")
		return truncate(value.replace(/\s+/g, " "), 180);
	return String(value);
}

function renderCellValue(cell, value) {
	const text = displayValue(value);
	if (!state.searchValue || value === null || typeof value === "object") {
		cell.textContent = text;
		return;
	}
	const query = state.searchValue.toLocaleLowerCase();
	const searchableText = text.toLocaleLowerCase();
	let cursor = 0;
	let matchIndex = searchableText.indexOf(query);
	if (matchIndex < 0) {
		cell.textContent = text;
		return;
	}
	while (matchIndex >= 0) {
		cell.append(document.createTextNode(text.slice(cursor, matchIndex)));
		cell.append(
			node(
				"mark",
				"search-match",
				text.slice(matchIndex, matchIndex + query.length),
			),
		);
		cursor = matchIndex + query.length;
		matchIndex = searchableText.indexOf(query, cursor);
	}
	cell.append(document.createTextNode(text.slice(cursor)));
}

function compactValue(value) {
	if (typeof value === "number")
		return Number.isInteger(value) ? String(value) : value.toPrecision(4);
	if (typeof value === "string") return truncate(value, 24);
	return truncate(JSON.stringify(value), 24);
}

function truncate(value, length) {
	return value.length > length ? `${value.slice(0, length - 1)}…` : value;
}

async function changeSort(columnName) {
	if (state.sortColumn === columnName) {
		state.sortDirection = state.sortDirection === "asc" ? "desc" : "asc";
	} else {
		state.sortColumn = columnName;
		state.sortDirection = "asc";
	}
	state.offset = 0;
	await loadRows();
}

function renderPager() {
	if (!state.page) return;
	const start = state.page.total === 0 ? 0 : state.page.offset + 1;
	const end = Math.min(
		state.page.offset + state.page.rows.length,
		state.page.total,
	);
	const resultLabel = state.searchValue || state.filterColumn ? " matches" : "";
	elements.pagerInfo.textContent = `${formatInteger(start)}-${formatInteger(end)} of ${formatInteger(state.page.total)}${resultLabel}`;
	elements.previousPage.disabled = state.offset === 0;
	elements.nextPage.disabled =
		state.offset + state.page.rows.length >= state.page.total;
}

function renderStructure() {
	const sections = [];
	const statsSection = node("section", "detail-section");
	statsSection.append(node("h3", "detail-heading", "Table statistics"));
	const statsGrid = node("div", "stats-grid");
	const stats = state.detail.stats;
	const statRows = [
		["Rows", formatInteger(stats.num_rows)],
		["Table bytes", formatBytes(stats.total_bytes)],
		["Indexes", formatInteger(stats.num_indices)],
		["Fragments", formatInteger(stats.fragment_stats?.num_fragments || 0)],
	];
	for (const [name, value] of statRows) {
		const item = node("div", "stat");
		item.append(node("div", "stat-name", name));
		item.append(node("div", "stat-value", value));
		statsGrid.append(item);
	}
	statsSection.append(statsGrid);
	sections.push(statsSection);

	const schemaSection = node("section", "detail-section");
	schemaSection.append(node("h3", "detail-heading", "Arrow schema"));
	schemaSection.append(
		detailTable(
			["Column", "Type", "Nullable", "Sortable", "Filterable"],
			state.detail.schema.map((field) => [
				field.name,
				field.type,
				field.nullable ? "yes" : "no",
				field.sortable ? "yes" : "no",
				field.filterable ? "yes" : "no",
			]),
		),
	);
	sections.push(schemaSection);

	const indexSection = node("section", "detail-section");
	indexSection.append(node("h3", "detail-heading", "Indexes"));
	if (state.detail.indexes.length === 0) {
		indexSection.append(node("div", "empty-state", "No indexes on this table"));
	} else {
		indexSection.append(
			detailTable(
				["Name", "Type", "Columns", "Indexed", "Unindexed", "Size"],
				state.detail.indexes.map((index) => [
					index.name,
					index.type,
					index.columns.join(", "),
					formatInteger(index.indexed_rows),
					formatInteger(index.unindexed_rows),
					formatBytes(index.bytes),
				]),
			),
		);
	}
	sections.push(indexSection);
	replaceChildren(elements.structurePanel, sections);
}

function detailTable(headers, rows) {
	const table = node("table", "detail-table");
	const thead = node("thead");
	const headerRow = node("tr");
	headers.forEach((header) => {
		headerRow.append(node("th", "", header));
	});
	thead.append(headerRow);
	table.append(thead);
	const tbody = node("tbody");
	rows.forEach((row) => {
		const tr = node("tr");
		row.forEach((value) => {
			tr.append(node("td", "", String(value)));
		});
		tbody.append(tr);
	});
	table.append(tbody);
	return table;
}

function activateTab(tabName) {
	state.tab = tabName;
	if (tabName !== "data") closeRowInspector();
	document.querySelectorAll(".tab").forEach((tab) => {
		const active = tab.dataset.tab === tabName;
		tab.classList.toggle("active", active);
		tab.setAttribute("aria-selected", String(active));
		tab.tabIndex = active ? 0 : -1;
	});
	elements.dataPanel.hidden = tabName !== "data";
	elements.structurePanel.hidden = tabName !== "structure";
	elements.dataSearchControls.hidden = tabName !== "data";
	syncSearchControls();
}

function openRow(row, rowNumber, trigger) {
	state.selectedRowNumber = rowNumber;
	state.inspectorTrigger = trigger;
	state.dialogJson = JSON.stringify(row, null, 2);
	elements.dialogTitle.textContent = `${state.table} / row ${rowNumber}`;
	elements.rawJson.textContent = state.dialogJson;
	elements.copyRow.textContent = "Copy";
	elements.rowInspector.hidden = false;
	document.querySelectorAll("tbody tr").forEach((tableRow) => {
		tableRow.classList.toggle(
			"selected",
			tableRow.dataset.rowNumber === String(rowNumber),
		);
	});
	document.querySelectorAll(".row-open").forEach((rowButton) => {
		rowButton.setAttribute("aria-expanded", String(rowButton === trigger));
	});
	elements.closeDialog.focus();
}

function closeRowInspector({ restoreFocus = false } = {}) {
	const trigger = state.inspectorTrigger;
	state.selectedRowNumber = null;
	state.inspectorTrigger = null;
	elements.rowInspector.hidden = true;
	document.querySelectorAll("tbody tr.selected").forEach((tableRow) => {
		tableRow.classList.remove("selected");
	});
	document
		.querySelectorAll(".row-open[aria-expanded='true']")
		.forEach((rowButton) => {
			rowButton.setAttribute("aria-expanded", "false");
		});
	if (restoreFocus && trigger?.isConnected) trigger.focus();
}

async function refreshAll() {
	elements.refreshButton.disabled = true;
	try {
		clearError();
		await loadOverview();
		if (state.table) {
			const encoded = encodeURIComponent(state.table);
			state.detail = await requestJson(`/api/tables/${encoded}`);
			await loadRows();
			renderCurrentTable();
		}
	} catch (error) {
		showError(error);
	} finally {
		elements.refreshButton.disabled = false;
	}
}

elements.tableSearch.addEventListener("input", renderTableList);
elements.rowSearch.addEventListener("input", () => {
	state.searchValue = elements.rowSearch.value.trim();
	state.offset = 0;
	elements.clearSearch.hidden = !state.searchValue;
	if (state.searchTimer) window.clearTimeout(state.searchTimer);
	state.searchTimer = window.setTimeout(loadRows, state.searchValue ? 450 : 0);
});
elements.rowSearch.addEventListener("keydown", (event) => {
	if (event.key !== "Enter") return;
	event.preventDefault();
	if (state.searchTimer) window.clearTimeout(state.searchTimer);
	loadRows();
});
elements.clearSearch.addEventListener("click", () => {
	if (state.searchTimer) window.clearTimeout(state.searchTimer);
	state.searchValue = "";
	state.offset = 0;
	elements.rowSearch.value = "";
	syncSearchControls();
	loadRows();
	elements.rowSearch.focus();
});
elements.filterToggle.addEventListener("click", () => {
	state.filterVisible = !state.filterVisible;
	if (state.filterVisible) populateFilterControls();
	syncSearchControls();
	if (state.filterVisible) elements.filterColumn.focus();
});
elements.closeFilter.addEventListener("click", () =>
	closeFilter({ restoreFocus: true }),
);
elements.refreshButton.addEventListener("click", refreshAll);
elements.themeToggle.addEventListener("click", () => {
	const theme =
		document.documentElement.dataset.theme === "dark" ? "light" : "dark";
	window.localStorage.setItem(THEME_STORAGE_KEY, theme);
	renderTheme(theme);
});
systemTheme.addEventListener("change", useSystemTheme);
elements.liveRefresh.addEventListener("change", () => {
	if (state.liveTimer) window.clearInterval(state.liveTimer);
	state.liveTimer = elements.liveRefresh.checked
		? window.setInterval(refreshAll, 5000)
		: null;
});
elements.filterColumn.addEventListener("change", () => {
	state.filterOperator = null;
	renderFilterOperators();
});
elements.filterOperator.addEventListener("change", renderFilterOperators);
elements.filterForm.addEventListener("submit", async (event) => {
	event.preventDefault();
	const column = elements.filterColumn.value;
	if (!column) {
		showError(new Error("Choose a filter column"));
		return;
	}
	state.filterColumn = column;
	state.filterOperator = elements.filterOperator.value;
	state.filterValue = ["is_null", "is_not_null"].includes(state.filterOperator)
		? null
		: elements.filterValue.value;
	state.offset = 0;
	closeFilter();
	await loadRows();
});
elements.clearFilter.addEventListener("click", async () => {
	state.filterColumn = null;
	state.filterOperator = null;
	state.filterValue = null;
	state.offset = 0;
	populateFilterControls();
	closeFilter();
	await loadRows();
});
elements.pageSize.addEventListener("change", async () => {
	state.limit = Number(elements.pageSize.value);
	state.offset = 0;
	await loadRows();
});
elements.previousPage.addEventListener("click", async () => {
	state.offset = Math.max(0, state.offset - state.limit);
	await loadRows();
});
elements.nextPage.addEventListener("click", async () => {
	state.offset += state.limit;
	await loadRows();
});
const tabs = [...document.querySelectorAll(".tab")];
tabs.forEach((tab, index) => {
	tab.addEventListener("click", () => activateTab(tab.dataset.tab));
	tab.addEventListener("keydown", (event) => {
		let nextIndex = null;
		if (event.key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
		if (event.key === "ArrowLeft")
			nextIndex = (index - 1 + tabs.length) % tabs.length;
		if (event.key === "Home") nextIndex = 0;
		if (event.key === "End") nextIndex = tabs.length - 1;
		if (nextIndex === null) return;
		event.preventDefault();
		tabs[nextIndex].focus();
		activateTab(tabs[nextIndex].dataset.tab);
	});
});
elements.closeDialog.addEventListener("click", () =>
	closeRowInspector({ restoreFocus: true }),
);
document.addEventListener("keydown", (event) => {
	if (
		(event.metaKey || event.ctrlKey) &&
		event.key.toLocaleLowerCase() === "k" &&
		state.tab === "data" &&
		!elements.rowSearch.disabled
	) {
		event.preventDefault();
		elements.rowSearch.focus();
		elements.rowSearch.select();
		return;
	}
	if (event.key !== "Escape") return;
	if (!elements.rowInspector.hidden) {
		closeRowInspector({ restoreFocus: true });
	} else if (state.filterVisible) {
		closeFilter({ restoreFocus: true });
	}
});
document.addEventListener("pointerdown", (event) => {
	if (
		!state.filterVisible ||
		elements.filterForm.contains(event.target) ||
		elements.filterToggle.contains(event.target)
	)
		return;
	closeFilter();
});
elements.copyRow.addEventListener("click", async () => {
	try {
		await navigator.clipboard.writeText(state.dialogJson);
		elements.copyRow.textContent = "Copied";
	} catch {
		elements.copyRow.textContent = "Copy failed";
	}
});

loadOverview({ selectInitial: true }).then(clearError).catch(showError);
