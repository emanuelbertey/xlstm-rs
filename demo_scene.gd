extends Control

var a:Array[Array]
var name_random:Array
var aux:PackedByteArray
var name_sed = "asdkgfjghtyeur"

# Called when the node enters the scene tree for the first time.
func _ready():
	var my_seed = "gkfkfdjgfjgvgtv".hash()
	seed(my_seed)
	
	
	if $Table.auto_reload:
		a.append(["This", "data", "loaded", "on"])
		a.append(["the", "_ready", "func"])
		$Table.table = a
	else:
		$auto_reload_test_btn.disabled = true


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta):
	$index_input.max_value = $Table.table.size() - 1


func _on_add_btn_pressed():
	aux = name_sed.to_utf8_buffer()#PackedByteArray.get_string_from_utf8
	name_random = Array(aux)
	name_random.shuffle()
	var b22 = randi()
	
	var array_long:Array = [array_to_string(name_random), "Array", "is", b22, "long"]
	var array_short:Array = ["And this", "is to", "short"]
	$Table.add_row(array_long)
	#$Table.add_row(array_short)
	


func _on_remove_last_btn_pressed():
	if $Table.remove_last_row():
		print("Last row was removed")
	else:
		print("Last row was not removed")


func _on_remove_row_at_btn_pressed():
	$Table.remove_row_at($index_input.value)


func _on_table_click_cell_data(cell:String):
	print("This is the Text of the selected cell: ", cell)


func _on_table_click_row(row:Array):
	print("This is the selected row: ", row)


func _on_table_click_cell_pos(pos:Vector2i):
	print("This is the Vector2i of the selected cell: ", pos)


func _on_table_click_row_index(index:int):
	print("This is the selected row index: ", index)


func _on_auto_reload_test_btn_pressed() -> void:
	a.append(["Spawn", "a", "new", "line"])


func _on_table_double_click(pos: Vector2i, key:Key) -> void:
	print("This is the Vector2i of the double clicked cell: ", pos)
	print("And the cell value ist: ", $Table.table[pos.y][pos.x])
	print("The pressed key was: ", key)



func array_to_string(arr: Array) -> String:
	var s = ""
	for i in arr:
		var char_ascii = char(i)
		s += char_ascii
	return s
@tool
extends EditorPlugin


func _enter_tree():
	# Initialization of the plugin goes here.
	add_custom_type("Table", "PanelContainer", preload("Table.gd"), preload("icon.svg"))


func _exit_tree():
	# Clean-up of the plugin goes here.
	remove_custom_type("Table")
@tool
extends PanelContainer

enum select_mode {CELL, ROW}

## Emitted when a cell is selected. [param cell] is the text.[br]
## [color=yellow]Important:[/color] it can only be used if [code]table_select_mode[/code] is set to [code]CELL[/code].
signal CLICK_CELL_DATA(cell:String)
## Emitted when a cell is selected. [param pos] is the position.[br]
## [color=yellow]Important:[/color] it can only be used if [code]table_select_mode[/code] is set to [code]CELL[/code].
signal CLICK_CELL_POS(pos:Vector2i)
## Emitted when when a row is selected. [param row] is the row as an array of strings.
signal CLICK_ROW(row:Array)
## Emitted when when a row is selected. [param index] is the index of the row, whereby the header row does not count and the first row of the table is 0.
signal CLICK_ROW_INDEX(index:int)
## Emitted when a cell is double clicked. [param pos] is the position of the cell.[br]
## [param key] is the type of activation.[br]
## Double-click is [code]KEY_NONE[/code][br]
## Enter key is [code]KEY_ENTER[/code][br]
## Space bar is [code]KEY_SPACE[/code][br]
signal DOUBLE_CLICK(pos:Vector2i, key:Key)

# user settings
@export var header_row:Array[String]: set = _set_header_row
## The custom minimum width of the header columns, at 0 the width is automatically adjusted to the text length
@export var header_width:Array[int]: set = _set_header_width
@export var table:Array[Array]: set = set_table
@export var table_select_mode:select_mode = select_mode.ROW: set = _set_table_select_mode
@export var table_allow_reselect:bool = false: set = _set_table_allow_reselect
## If active, the _ready function checks whether 'table' has changed, if yes, the table is reloaded
@export var auto_reload:bool = false
@export_group("Header")
@export var header_stylebox_normal:StyleBox: set = _set_header_stylebox_normal
@export var header_stylebox_pressed:StyleBox: set = _set_header_stylebox_pressed
@export var header_stylebox_hover:StyleBox: set = _set_header_stylebox_hover
@export_group("Table")
@export var background_stylebox:StyleBox: set = _set_stylebox_background
@export_group("Font")
@export var header_font:Font: set = _set_header_font
@export var header_font_color:Color: set = _set_header_font_color
## Sets the font size of the table header, if 0 then the default size is used
@export_range(0, 1, 1, "or_greater") var header_font_size:int: set = _set_header_font_size
@export var table_font:Font: set = _set_table_font
@export var table_font_color:Color: set = _set_table_font_color
## Sets the font size of the table, if 0 then the default size is used
@export_range(0, 1, 1, "or_greater") var table_font_size:int: set = _set_table_font_size



const TableContainer = preload("res://addons/godot_tree_table/TableContainer.gd")
var preload_tableContainer:PackedScene = preload("res://addons/godot_tree_table/TableContainer.tscn")

var tableContainer:TableContainer


func _init() -> void:
	_init_tree()


func _init_tree() -> void:
	tableContainer = preload_tableContainer.instantiate()
	self.add_child(tableContainer, true)
	
	tableContainer._init_tree()


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	tableContainer.CLICK_CELL_DATA.connect(_on_click_cell_data)
	tableContainer.CLICK_CELL_POS.connect(_on_click_cell_pos)
	tableContainer.CLICK_ROW.connect(_on_click_row)
	tableContainer.CLICK_ROW_INDEX.connect(_on_click_row_index)
	tableContainer.DOUBLE_CLICK.connect(_on_double_click)


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta) -> void:
	if auto_reload:
		tableContainer.auto_reload(table)


## Sets the entire table to the passed array, shortens or expands the inner arrays if necessary
func set_table(new_table:Array[Array]) -> void:
	for row:Array in new_table:
		var row_columns:int = row.size()
		var header_columns:int = header_row.size()
		if row_columns > header_columns:
			row.resize(header_columns)
		elif row_columns < header_columns:
			row.resize(header_columns)
			for i:int in row.size():
				if typeof(row[i]) == TYPE_NIL:
					row[i] = "----"
	table = new_table
	
	tableContainer.set_original_table(table)
	tableContainer.set_table(table, header_row.size())

## Adds the passed array to the table, shortens or extends it if necessary 
func add_row(new_row:Array) -> void:
	table.append(new_row)
	set_table(table)

## Deletes the last row of the table, true if it deleted something, otherwise false, returns no errors
func remove_last_row() -> bool:
	if table.size() == 0:
		return false
	table.remove_at(table.size() - 1)
	set_table(table)
	return true

## Deletes the row on the passed index, counted from the end if the index is negative
func remove_row_at(index:int) -> void:
	if index >= table.size():
		push_error("Error: Index pos = %d is out of bounds (size() = %d)." % [index, table.size()])
		return
	if index < 0:
		table.remove_at(table.size() + index)
	else:
		table.remove_at(index)
	set_table(table)

## Reloads the table
func reload_table() -> void:
	tableContainer.reload_table(table)


# -- Inernal funtions --
func _set_header_row(value:Array[String]) -> void:
	header_row = value
	tableContainer.set_header(header_row)
	
	header_width.resize(header_row.size())


func _set_stylebox_background(value:StyleBox) -> void:
	background_stylebox = value
	tableContainer.set_stylebox_background(background_stylebox)


func _set_header_stylebox_normal(value:StyleBox) -> void:
	header_stylebox_normal = value
	tableContainer.set_header_stylebox_normal(header_stylebox_normal)


func _set_header_stylebox_pressed(value:StyleBox) -> void:
	header_stylebox_pressed = value
	tableContainer.set_header_stylebox_pressed(header_stylebox_pressed)


func _set_header_stylebox_hover(value:StyleBox) -> void:
	header_stylebox_hover = value
	tableContainer.set_header_stylebox_hover(header_stylebox_hover)


func _set_header_width(value:Array[int]) -> void:
	header_width = value
	
	for i:int in header_width.size():
		tableContainer.set_header_width(i, header_width[i])


func _set_header_font(value:Font) -> void:
	header_font = value
	tableContainer.set_header_font(header_font)


func _set_header_font_color(value:Color) -> void:
	header_font_color = value
	tableContainer.set_header_font_color(header_font_color)


func _set_header_font_size(value:int) -> void:
	header_font_size = value
	tableContainer.set_header_font_size(header_font_size)


func _set_table_font(value:Font) -> void:
	table_font = value
	tableContainer.set_table_font(table_font)


func _set_table_font_color(value:Color) -> void:
	table_font_color = value
	tableContainer.set_table_font_color(table_font_color)


func _set_table_font_size(value:int) -> void:
	table_font_size = value
	tableContainer.set_table_font_size(table_font_size)


func _set_table_select_mode(value:select_mode) -> void:
	table_select_mode = value
	
	match table_select_mode:
		select_mode.CELL:
			tableContainer.set_select_mode(true)
		select_mode.ROW:
			tableContainer.set_select_mode(false)


func _set_table_allow_reselect(value:bool) -> void:
	table_allow_reselect = value
	tableContainer.set_allow_reselect(table_allow_reselect)


# -- signal functions --
func _on_click_cell_data(result:String) -> void:
	CLICK_CELL_DATA.emit(result)

func _on_click_row(result:Array) -> void:
	CLICK_ROW.emit(result)

func _on_click_cell_pos(result:Vector2i) -> void:
	CLICK_CELL_POS.emit(result)

func _on_click_row_index(result:int) -> void:
	CLICK_ROW_INDEX.emit(result)

func _on_double_click(result:Vector2i, key:Key) -> void:
	DOUBLE_CLICK.emit(result, key)



@tool
extends Control

signal CLICK_CELL_DATA(cell:String)
signal CLICK_CELL_POS(pos:Vector2i)
signal CLICK_ROW(row:Array)
signal CLICK_ROW_INDEX(index:int)
signal DOUBLE_CLICK(pos:Vector2i, key:Key)


var tree:Tree
var tree_root:TreeItem
var background:Panel

var original_table:Array[Array]
var sort_mode_ascending:bool = true


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	tree.column_title_clicked.connect(on_column_title_clicked)
	
	tree.cell_selected.connect(get_cell_data)
	tree.cell_selected.connect(get_cell_pos_selected_cell)
	tree.item_selected.connect(get_row_data)
	tree.item_selected.connect(get_row_index)
	tree.item_activated.connect(get_cell_pos_double_click)


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta) -> void:
	pass


func _init_tree() -> void:
	tree = $Background/ScrollContainer/Tree
	background = $Background


func set_header(header_row:Array[String]) -> void:
	if header_row.size() < 1:
		tree.columns = 1
		tree.set_column_title(0, "")
		return
	
	tree.columns = header_row.size()
	
	for i:int in header_row.size():
		tree.set_column_title(i, header_row[i])
		tree.set_column_title_alignment(i, HORIZONTAL_ALIGNMENT_LEFT)


func set_table(table:Array[Array], header_size:int) -> void:
	tree.clear()
	tree_root = tree.create_item()
	tree.hide_root = true
	for row:int in table.size():
		var item:TreeItem = tree.create_item(tree_root)
		for column:int in header_size:
			item.set_text(column, str(table[row][column]))


func reload_table(table:Array[Array]) -> void:
	set_table(table, tree.columns)


func auto_reload(table:Array[Array]) -> void:
	if table != original_table:
		set_table(table, tree.columns)


func set_original_table(table:Array[Array]) -> void:
	original_table = table.duplicate(true)


func set_stylebox_background(stylebox:StyleBox) -> void:
	if stylebox:
		background.add_theme_stylebox_override("panel", stylebox)
		return
	background.remove_theme_stylebox_override("panel")


func set_header_stylebox_normal(stylebox:StyleBox) -> void:
	if stylebox:
		tree.add_theme_stylebox_override("title_button_normal", stylebox)
		return
	tree.remove_theme_stylebox_override("title_button_normal")


func set_header_stylebox_pressed(stylebox:StyleBox) -> void:
	if stylebox:
		tree.add_theme_stylebox_override("title_button_pressed", stylebox)
		return
	tree.remove_theme_stylebox_override("title_button_pressed")


func set_header_stylebox_hover(stylebox:StyleBox) -> void:
	if stylebox:
		tree.add_theme_stylebox_override("title_button_hover", stylebox)
		return
	tree.remove_theme_stylebox_override("title_button_hover")


func set_header_width(column:int, width:int) -> void:
	tree.set_column_custom_minimum_width(column, width)


func set_header_font(font:Font) -> void:
	if font:
		tree.add_theme_font_override("title_button_font", font)
		return
	tree.remove_theme_font_override("title_button_font")


func set_header_font_color(color:Color) -> void:
	if color:
		tree.add_theme_color_override("title_button_color", color)
		return
	tree.remove_theme_color_override("title_button_color")


func set_header_font_size(size:int) -> void:
	if size and size > 0:
		tree.add_theme_font_size_override("title_button_font_size", size)
		return
	tree.remove_theme_font_size_override("title_button_font_size")


func set_table_font(font:Font) -> void:
	if font:
		tree.add_theme_font_override("font", font)
		return
	tree.remove_theme_font_override("font")


func set_table_font_color(color:Color) -> void:
	if color:
		tree.add_theme_color_override("font_color", color)
		return
	tree.remove_theme_color_override("font_color")


func set_table_font_size(size:int) -> void:
	if size and size > 0:
		tree.add_theme_font_size_override("font_size", size)
		return
	tree.remove_theme_font_size_override("font_size")


func set_select_mode(mode:bool) -> void:
	if mode:
		tree.select_mode = Tree.SELECT_SINGLE
	else:
		tree.select_mode = Tree.SELECT_ROW


func set_allow_reselect(reselect:bool) -> void:
	tree.allow_reselect = reselect


# -- signal functions --
func get_cell_data() -> void:
	CLICK_CELL_DATA.emit(tree.get_selected().get_text(tree.get_selected_column()))


func get_cell_pos_selected_cell() -> void:
	var result:Vector2i = Vector2i(-1, -1)
	result.x = tree.get_selected_column()
	result.y = tree.get_root().get_children().find(tree.get_selected())
	
	CLICK_CELL_POS.emit(result)


func get_row_data() -> void:
	var result:Array
	var sel_item:TreeItem = tree.get_selected()
	for i:int in tree.columns:
		result.append(sel_item.get_text(i))
	
	CLICK_ROW.emit(result)


func get_row_index() -> void:
	var result:int = -1
	result = tree.get_root().get_children().find(tree.get_selected())
	
	CLICK_ROW_INDEX.emit(result)


func get_cell_pos_double_click() -> void:
	var result:Vector2i = Vector2i(-1, -1)
	var key:Key = KEY_NONE
	if Input.is_key_pressed(KEY_ENTER):
		key = KEY_ENTER
	if Input.is_key_pressed(KEY_SPACE):
		key = KEY_SPACE
	result.x = tree.get_selected_column()
	result.y = tree.get_root().get_children().find(tree.get_selected())
	
	DOUBLE_CLICK.emit(result, key)


func on_column_title_clicked(column:int, mouse_button_index:int) -> void:
	if mouse_button_index == MOUSE_BUTTON_LEFT or mouse_button_index == MOUSE_BUTTON_RIGHT:
		var sorted_table:Array[Array] = original_table.duplicate(true)
		match mouse_button_index:
			MOUSE_BUTTON_LEFT:
				if sort_mode_ascending:
					sorted_table.sort_custom(custom_sorter_ascending.bind(column))
				else:
					sorted_table.sort_custom(custom_sorter_descending.bind(column))
				sort_mode_ascending = !sort_mode_ascending
				set_table(sorted_table, tree.columns)
			MOUSE_BUTTON_RIGHT:
				sort_mode_ascending = true
				set_table(original_table, tree.columns)


# -- custom sorter --
static func custom_sorter_ascending(a, b, column:int) -> bool:
	var tipo_a = typeof(a[column])
	var tipo_b = typeof(b[column])
	
	if tipo_a == TYPE_INT :
		a[column] = str(a[column])
	if tipo_b == TYPE_INT:
		b[column] = str(b[column])
		
	if a[column] == "----":
		return false
	if a[column] <= b[column]:
		return true
	return false

static func custom_sorter_descending(a, b, column:int) -> bool:
	var tipo_a = typeof(a[column])
	var tipo_b = typeof(b[column])
	
	if tipo_a == TYPE_INT :
		a[column] = str(a[column])
	if tipo_b == TYPE_INT:
		b[column] = str(b[column])
		
	if a[column] == "----":
		return false
	if a[column] >= b[column]:
		return true
	return false


extends Control


@export_enum("YES_ARRAY:0", "NOT_NADA:1") var PRUEBA: int = 0

@export_enum("FASTLZ:0", "DEFLATE:1", "ZSTD:2", "GZIP:3" ) var MODE_COMPRESSION: int = 0# Called when the node enters the scene tree for the first time.
var cadena = "prueba12hfgbhhyghkghnjkmkbgfyjknhhhjnh"
var string_length
var bytes


func _ready() -> void:
	# tu cadena

	string_length = cadena.length()

# obtener matriz de bytes
	bytes = cadena.to_utf8_buffer()
	prints(bytes.size(), string_length, "prueba")
	if PRUEBA == 0:
		for i in bytes:
			prints(i)

# compresión, usando Deflate
	var comprimido = bytes.compress(MODE_COMPRESSION)
	prints(comprimido.size(), "comprimido")
	prints(string_length)
	if PRUEBA == 0:
		for i in bytes:
			prints(i)
# descompresión usando Deflate, con longitud de salida conocida


	var cl = compress.new()
	prints(cl.decompress(comprimido, 0, string_length ))
	prints(cl.compress(cadena, 0))



	var descomprimido = comprimido.decompress(string_length, MODE_COMPRESSION)
	var cadena_decomprimida = descomprimido.get_string_from_utf8()
	prints(cadena_decomprimida)

## descompresión mediante Deflate, con longitud de salida desconocida
	#var decomp_dynamic = comprimido.decompress_dynamic(-1, MODE_COMPRESSION)
	#cadena_decomprimida = decomp_dynamic.get_string_from_utf8()
	#prints(cadena_decomprimida)
	#pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass


extends Control


@export_enum("YES_ARRAY:0", "NOT_NADA:1") var PRUEBA: int = 0
@export_enum("DEFLATE:1", "GZIP:3") var MODE_COMPRESSION: int = 1

var cadena = "dde las manzanas nace fruta verde y rojaeu"
var string_length
var bytes
var bite: PackedFloat32Array
var bite2: PackedByteArray


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	# tu cadena

	string_length = cadena.length()
	bite2 = bite.to_byte_array()

	prints(bite2, "bite 2 nulo sin nada ")

# obtener matriz de bytes
	bite2 = cadena.to_utf8_buffer()
	#var ir = bite2.size() / 4
	#bite2.resize(ir)
	bite = bite2.to_float32_array()# no use anda mal
	prints(bite, "prins bite con datos de bite2 prueva 1")
	var bitearr: Array = Array(bite2)
	bite = PackedFloat32Array(bitearr)
	prints(bite, "prins bite con datos de bite2 desde array a array32float prueva 2")
	bite2 = PackedByteArray(bitearr)
	prints(bite2, "prins bite2 packarray de un array prueva 3")
	bite2 = bite.to_byte_array()
	prints(bite2, "prueba packarray de un packarray32float")
	bitearr = Array(bite)
	prints(bitearr, "prueba array de un packarray32float")
	bitearr = Array(bite)


	prints("size array ", bite2.size())
	prints(bite2, "bite 2 luego de asignar string")
	bytes = cadena.to_utf8_buffer()
	prints(bytes, "prints de bytes con string para comparara")



	if PRUEBA == 0:
		for i in bytes:
			prints(i)


# compresión, usando Deflate
	var comprimido = bytes.compress(MODE_COMPRESSION)
	prints(comprimido.size(), " uanto pesa comprimido")
	prints(string_length, " uanto pesa sin comprimir")
	prints(comprimido, " dato comprimido brutioo")

# descompresión usando Deflate, con longitud de salida conocida
	var descomprimido = comprimido.decompress(string_length, MODE_COMPRESSION)
	var cadena_decomprimida = descomprimido.get_string_from_utf8()
	prints(cadena_decomprimida, "  no dinamio on get string")

# descompresión mediante Deflate, con longitud de salida desconocida
	var decomp_dynamic = comprimido.decompress_dynamic(-1, MODE_COMPRESSION)
	cadena_decomprimida = decomp_dynamic.get_string_from_utf8()
	prints(decomp_dynamic, " dinamico sin getstring")
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass

extends Control


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	#var prueva = compress.new()
	#add_child(prjghfjdhjlllll
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:

	pass

extends Control

@onready var ncar = $autos
var nodes = []
var total_entrenado = 0 
var red_neuronal
var epoca = 30
# Datos de 121 carros -> [Antigüedad, costo de salida al mercado]

var x : Array = [[0.0, 1.0], [0.1, 1.0], [0.2, 1.0], [0.3, 1.0], [0.4, 1.0],
	[0.5, 1.0], [0.6, 1.0], [0.7, 1.0], [0.8, 1.0], [0.9, 1.0],
	[1.0, 1.0], [0.0, 0.9], [0.1, 0.9], [0.2, 0.9], [0.3, 0.9],
	[0.4, 0.9], [0.5, 0.9], [0.6, 0.9], [0.7, 0.9], [0.8, 0.9],
	[0.9, 0.9], [1.0, 0.9], [0.0, 0.8], [0.1, 0.8], [0.2, 0.8],
	[0.3, 0.8], [0.4, 0.8], [0.5, 0.8], [0.6, 0.8], [0.7, 0.8],
	[0.8, 0.8], [0.9, 0.8], [1.0, 0.8], [0.0, 0.7], [0.1, 0.7],
	[0.2, 0.7], [0.3, 0.7], [0.4, 0.7], [0.5, 0.7], [0.6, 0.7],
	[0.7, 0.7], [0.8, 0.7], [0.9, 0.7], [1.0, 0.7], [0.0, 0.6],
	[0.1, 0.6], [0.2, 0.6], [0.3, 0.6], [0.4, 0.6], [0.5, 0.6],
	[0.6, 0.6], [0.7, 0.6], [0.8, 0.6], [0.9, 0.6], [1.0, 0.6],
	[0.0, 0.5], [0.1, 0.5], [0.2, 0.5], [0.3, 0.5], [0.4, 0.5],
	[0.5, 0.5], [0.6, 0.5], [0.7, 0.5], [0.8, 0.5], [0.9, 0.5],
	[1.0, 0.5], [0.0, 0.4], [0.1, 0.4], [0.2, 0.4], [0.3, 0.4],
	[0.4, 0.4], [0.5, 0.4], [0.6, 0.4], [0.7, 0.4], [0.8, 0.4],
	[0.9, 0.4], [1.0, 0.4], [0.0, 0.3], [0.1, 0.3], [0.2, 0.3],
	[0.3, 0.3], [0.4, 0.3], [0.5, 0.3], [0.6, 0.3], [0.7, 0.3],
	[0.8, 0.3], [0.9, 0.3], [1.0, 0.3], [0.0, 0.2], [0.1, 0.2],
	[0.2, 0.2], [0.3, 0.2], [0.4, 0.2], [0.5, 0.2], [0.6, 0.2],
	[0.7, 0.2], [0.8, 0.2], [0.9, 0.2], [1.0, 0.2], [0.0, 0.1],
	[0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.4, 0.1], [0.5, 0.1],
	[0.6, 0.1], [0.7, 0.1], [0.8, 0.1], [0.9, 0.1], [1.0, 0.1],
	[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0],
	[0.5, 0.0], [0.6, 0.0], [0.7, 0.0], [0.8, 0.0], [0.9, 0.0],
	[1.0, 0.0]]

# 0 : normal    1 : coleccionable

var y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
	0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
	0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
	0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
	0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
	0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
	0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
	0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
 #Los datos x y y no se incluyen aquí porque ya los tienes definidos.

func _ready():
	
	
	var long = 0
	for a in range(x.size()):
		for l in range(1):
			if y[long] == 1:
				var car = load("res://car/coleccion.tscn").instantiate()
				ncar.add_child(car)
				nodes.append(car)
			else:
				var car = load("res://car/comun.tscn").instantiate()
				ncar.add_child(car)
				nodes.append(car)
		long += 1
	
	
	
	
	
	
	
	
	randomize()
	#var x = [[0.0, 1.0], [0.1, 1.0], ...]  # Reemplaza con tu array de datos
	#var y = [0, 0, 0, ...]  # Reemplaza con tu array de etiquetas
	var n = randi()  % 100
	red_neuronal = RedNeuronal.new(x, y , epoca)
	prints(red_neuronal.clasificacion(x[n][0], x[n][1]))
	#red_neuronal.entrenamiento()
	#print("Entrenamiento completado")
	#
	#
	#for k in range(100):
		#var forward = red_neuronal.clasificacion(x[k][0], x[k][1])
		#var presicion = y[k]
		#var redondear = round(forward )
		#if redondear == presicion :
			#prints("la red acerto felicidades :)  redondeo ", redondear, " el dato :" ,y[k] , " la salida bruta : " ,forward)
		#else:
			#push_warning("la red erro prueba entrenar mas :)  redondeo ", redondear, " el dato :" ,y[k] , " la salida bruta : " ,forward)
			#pass


func prueba(nn):
	var error = 0
	for k in range(121):
		var forward = nn.clasificacion(x[k][0], x[k][1])
		var presicion = y[k]
		var redondear = round(forward )
		if redondear == presicion :
			nodes[k].valor = presicion
			nodes[k].obtenido = forward
			nodes[k].mal.visible = false
			nodes[k].bien.visible = true
			prints("la red acerto felicidades :)  redondeo ", redondear, " el dato :" ,y[k] , " la salida bruta : " ,forward)
		else:
			nodes[k].valor = presicion
			nodes[k].obtenido = forward
			nodes[k].mal.visible = true
			nodes[k].bien.visible = false
			push_warning("la red erro prueba entrenar mas :)  redondeo ", redondear, " el dato :" ,y[k] , " la salida bruta : " ,forward)
			error += 1
	$Label.text = " total de errores de la prueba es de: ( " + str(error) + " ) , la precicion maxima es 6 "









class RedNeuronal:
	var x
	var y
	var pesos1 = []
	var sesgos1 = []
	var pesos2 = []
	var sesgos2 = []
	var epoca = 0
	func _init(x, y, epoca):
		self.x = x
		self.y = y
		self.epoca = epoca
		# Inicialización de los pesos y sesgos
		for i in range(4):
			self.pesos1.append(randf())
		for i in range(2):
			self.sesgos1.append(randf())
		for i in range(2):
			self.pesos2.append(randf())
		self.sesgos2.append(randf())

	func entrenamiento(tasa_aprendizaje = 0.4, epocas = epoca):
		var timer_local = Time.get_ticks_usec()
		for k in range(epocas):
			
			var error = 0.0
			for i in range(self.x.size()):
				# Entradas a las neuronas sigmoides ocultas
				var suma_o1 = self.x[i][0] * self.pesos1[0] + self.x[i][1] * self.pesos1[2] + self.sesgos1[0]
				var suma_o2 = self.x[i][0] * self.pesos1[1] + self.x[i][1] * self.pesos1[3] + self.sesgos1[1]
				
				# Salidas de las neuronas sigmoides ocultas
				var salida_o1 = 1.0 / (1.0 + exp(-suma_o1))
				var salida_o2 = 1.0 / (1.0 + exp(-suma_o2))
				
				# Entrada de la neurona sigmoide de la capa de salida
				var suma_s = salida_o1 * self.pesos2[0] + salida_o2 * self.pesos2[1] + self.sesgos2[0]
				
				# Salida de la red neuronal
				var y_gorro = 1.0 / (1.0 + exp(-suma_s))
				#var y_gorro = tanh(suma_s)
				#var y_gorro = 1 * (1 - suma_s)
				#var y_gorro = max(0.001, suma_s) # relu
				#var y_gorro = log(1 + exp(suma_s)) # softplus

				# Cálculo del error cuassdrático
				error += (0.5) * pow(self.y[i] - y_gorro, 2)
				
				# Cálculo de los gradientes
				var gradiente_p21 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * salida_o1
				var gradiente_p22 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * salida_o2
				var gradiente_sesgo21 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * 1.0

				var gradiente_p11 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * self.pesos2[0] * (salida_o1 * (1 - salida_o1)) * self.x[i][0]
				var gradiente_p13 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * self.pesos2[0] * (salida_o1 * (1 - salida_o1)) * self.x[i][1]
				var gradiente_sesgo11 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * self.pesos2[0] * (salida_o1 * (1 - salida_o1)) * 1.0

				var gradiente_p12 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * self.pesos2[1] * (salida_o2 * (1 - salida_o2)) * self.x[i][0]
				var gradiente_p14 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * self.pesos2[1] * (salida_o2 * (1 - salida_o2)) * self.x[i][1]
				var gradiente_sesgo12 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * self.pesos2[1] * (salida_o2 * (1 - salida_o2)) * 1.0

				# Actualización de los pesos
				for l in range(1):
					self.pesos1[0] -= tasa_aprendizaje * gradiente_p11
					self.pesos1[1] -= tasa_aprendizaje * gradiente_p12
					self.pesos1[2] -= tasa_aprendizaje * gradiente_p13
					self.pesos1[3] -= tasa_aprendizaje * gradiente_p14
					self.sesgos1[0] -= tasa_aprendizaje * gradiente_sesgo11
					self.sesgos1[1] -= tasa_aprendizaje * gradiente_sesgo12
					self.pesos2[0] -= tasa_aprendizaje * gradiente_p21
					self.pesos2[1] -= tasa_aprendizaje * gradiente_p22
					self.sesgos2[0] -= tasa_aprendizaje * gradiente_sesgo21
		prints(" el entrenamiento demoro  (" ,Time.get_ticks_usec() - timer_local,") usec en :" , epoca , " entenamientos")
			#print(error)

	func clasificacion(x1, x2):
		var suma_o1 = x1 * self.pesos1[0] + x2 * self.pesos1[2] + self.sesgos1[0]
		var suma_o2 = x1 * self.pesos1[1] + x2 * self.pesos1[3] + self.sesgos1[1]
		var salida_o1 = 1.0 / (1.0 + exp(-suma_o1))
		var salida_o2 = 1.0 / (1.0 + exp(-suma_o2))
		var suma_s = salida_o1 * self.pesos2[0] + salida_o2 * self.pesos2[1] + self.sesgos2[0]
		var y_gorro = 1.0 / (1.0 + exp(-suma_s))
		return y_gorro#round(y_gorro)





func _on_probar_pressed() -> void:
	if red_neuronal == null:
		prints("error no iniciado")
	prueba(red_neuronal)
	pass # Replace with function body.


func _on_entrenar_pressed() -> void:
	if red_neuronal == null:
		prints("error no iniciado")
	red_neuronal.entrenamiento()
	total_entrenado += epoca 
	$Label2.text = " total de entrenamientos : ( " + str(total_entrenado) + " ) "
	prueba(red_neuronal)
	pass # Replace with function body.

extends Control

var x = [[0.0,0.0], [1.0,0.0], [0.0,1.0], [1.0,1.0]]
var y = [0, 1, 1, 0]


func _ready():
	randomize()
	#var x = [[0.0, 1.0], [0.1, 1.0], ...]  # Reemplaza con tu array de datos
	#var y = [0, 0, 0, ...]  # Reemplaza con tu array de etiquetas
	var n = randi()  % 4
	var red_neuronal = RedNeuronal.new(x, y)
	prints(red_neuronal.clasificacion(x[n][0], x[n][1]))
	red_neuronal.entrenamiento()
	print("Entrenamiento completado")
	
	
	for k in range(4):
	
		var forward = red_neuronal.clasificacion(x[k][0], x[k][1])
		var presicion = y[k]
		var redondear = round(forward )
		if redondear == presicion :
			prints(k,"la red acerto felicidades :)  redondeo ", redondear, " el dato :" ,y[k] , " la salida bruta : " ,forward)
		else:
			push_warning("la red erro prueba entrenar mas :)  redondeo ", redondear, " el dato :" ,y[k] , " la salida bruta : " ,forward)
			pass





class RedNeuronal:
	var x
	var y
	var pesos1 = []
	var sesgos1 = []
	var pesos2 = []
	var sesgos2 = []

	func _init(x, y):
		self.x = x
		self.y = y

		# Inicialización de los pesos y sesgos
		for i in range(4):
			self.pesos1.append(randf())
		for i in range(2):
			self.sesgos1.append(randf())
		for i in range(2):
			self.pesos2.append(randf())
		self.sesgos2.append(randf())

	func entrenamiento(tasa_aprendizaje = 0.4, epocas = 4000):
		for k in range(epocas ):
			var error = 0.0
			for i in range(self.x.size()):
				# Entradas a las neuronas sigmoides ocultas
				var suma_o1 = self.x[i][0] * self.pesos1[0] + self.x[i][1] * self.pesos1[2] + self.sesgos1[0]
				var suma_o2 = self.x[i][0] * self.pesos1[1] + self.x[i][1] * self.pesos1[3] + self.sesgos1[1]
				
				# Salidas de las neuronas sigmoides ocultas
				var salida_o1 = 1.0 / (1.0 + exp(-suma_o1))
				var salida_o2 = 1.0 / (1.0 + exp(-suma_o2))
				
				# Entrada de la neurona sigmoide de la capa de salida
				var suma_s = salida_o1 * self.pesos2[0] + salida_o2 * self.pesos2[1] + self.sesgos2[0]
				
				# Salida de la red neuronal
				var y_gorro = 1 / (1 + exp(-suma_s))

				# Cálculo del error cuadrático
				error += (0.5) * pow(self.y[i] - y_gorro, 2)
				
				# Cálculo de los gradientes
				var gradiente_p21 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * salida_o1
				var gradiente_p22 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * salida_o2
				var gradiente_sesgo21 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * 1.0

				var gradiente_p11 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * self.pesos2[0] * (salida_o1 * (1 - salida_o1)) * self.x[i][0]
				var gradiente_p13 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * self.pesos2[0] * (salida_o1 * (1 - salida_o1)) * self.x[i][1]
				var gradiente_sesgo11 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * self.pesos2[0] * (salida_o1 * (1 - salida_o1)) * 1.5

				var gradiente_p12 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * self.pesos2[1] * (salida_o2 * (1 - salida_o2)) * self.x[i][0]
				var gradiente_p14 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * self.pesos2[1] * (salida_o2 * (1 - salida_o2)) * self.x[i][1]
				var gradiente_sesgo12 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * self.pesos2[1] * (salida_o2 * (1 - salida_o2)) * 1.5

				# Actualización de los pesos
				self.pesos1[0] -= tasa_aprendizaje * gradiente_p11
				self.pesos1[1] -= tasa_aprendizaje * gradiente_p12
				self.pesos1[2] -= tasa_aprendizaje * gradiente_p13
				self.pesos1[3] -= tasa_aprendizaje * gradiente_p14
				self.sesgos1[0] -= tasa_aprendizaje * gradiente_sesgo11
				self.sesgos1[1] -= tasa_aprendizaje * gradiente_sesgo12
				self.pesos2[0] -= tasa_aprendizaje * gradiente_p21
				self.pesos2[1] -= tasa_aprendizaje * gradiente_p22
				self.sesgos2[0] -= tasa_aprendizaje * gradiente_sesgo21
			
			#print(error)

	func clasificacion(x1, x2):
		var suma_o1 = x1 * self.pesos1[0] + x2 * self.pesos1[2] + self.sesgos1[0]
		var suma_o2 = x1 * self.pesos1[1] + x2 * self.pesos1[3] + self.sesgos1[1]
		var salida_o1 = 1.0 / (1.0 + exp(-suma_o1))
		var salida_o2 = 1.0 / (1.0 + exp(-suma_o2))
		var suma_s = salida_o1 * self.pesos2[0] + salida_o2 * self.pesos2[1] + self.sesgos2[0]
		var y_gorro = 1.0 / (1.0 + exp(-suma_s))
		return y_gorro#round(y_gorro)

extends TextureRect
@onready var mal = $mal
@onready var bien = $bien
var valor = 1
var obtenido = 0


func _on_button_pressed() -> void:
	prints("mi valor es : ",valor , " - obtuve el valor : ", obtenido )
	pass # Replace with function body.

extends TextureRect
@onready var mal = $mal
@onready var bien = $bien
var valor = 1
var obtenido = 0


func _on_button_pressed() -> void:
	prints("mi valor es : ",valor , " - obtuve el valor : ", obtenido )
	pass # Replace with function body.


extends RefCounted

class_name ActivationFunction

var activation_func : Callable
var derivative_func : Callable

func _init(types: String = "Sigmoid") -> void:
	# Por defecto usa la función Sigmoid
	if types == "Sigmoid":
		activation_func = Callable(self, "_sigmoid")
		derivative_func = Callable(self, "_dsigmoid")
	else:
		activation_func = Callable(self, "_sigmoid")
		derivative_func = Callable(self, "_dsigmoid")

func _sigmoid(x: float) -> float:
	return 1.0 / (1.0 + exp(-x))

# Derivada de la función Sigmoid en términos de salida (y)
func _dsigmoid(y: float) -> float:
	return y * (1.0 - y)
extends Node
class_name MLP

var armado : Array = []
var neurona : Array = []
var structure : Array = []
var pesos : PackedFloat32Array = []
var sesgos : PackedFloat32Array = []
var total_neurona : int = 0




func _init(structure : Array , sesgo : bool = false) -> void:
	
	for neuronas in range(structure.size()):
		total_neurona += structure[neuronas]
	
	#for armar in range(structure.size()):
		#if armar < structure.size():
			#armado
		
	prints(structure , " estructura ")
	prints("total neuronas " , total_neurona)
	
	pass
extends Control
#
## Datos de 121 carros -> [Antigüedad, costo de salida al mercado]
#
#x = np.array([[0.0, 1.0], [0.1, 1.0], [0.2, 1.0], [0.3, 1.0], [0.4, 1.0],
			  #[0.5, 1.0], [0.6, 1.0], [0.7, 1.0], [0.8, 1.0], [0.9, 1.0],
			  #[1.0, 1.0], [0.0, 0.9], [0.1, 0.9], [0.2, 0.9], [0.3, 0.9],
			  #[0.4, 0.9], [0.5, 0.9], [0.6, 0.9], [0.7, 0.9], [0.8, 0.9],
			  #[0.9, 0.9], [1.0, 0.9], [0.0, 0.8], [0.1, 0.8], [0.2, 0.8],
			  #[0.3, 0.8], [0.4, 0.8], [0.5, 0.8], [0.6, 0.8], [0.7, 0.8],
			  #[0.8, 0.8], [0.9, 0.8], [1.0, 0.8], [0.0, 0.7], [0.1, 0.7],
			  #[0.2, 0.7], [0.3, 0.7], [0.4, 0.7], [0.5, 0.7], [0.6, 0.7],
			  #[0.7, 0.7], [0.8, 0.7], [0.9, 0.7], [1.0, 0.7], [0.0, 0.6],
			  #[0.1, 0.6], [0.2, 0.6], [0.3, 0.6], [0.4, 0.6], [0.5, 0.6],
			  #[0.6, 0.6], [0.7, 0.6], [0.8, 0.6], [0.9, 0.6], [1.0, 0.6],
			  #[0.0, 0.5], [0.1, 0.5], [0.2, 0.5], [0.3, 0.5], [0.4, 0.5],
			  #[0.5, 0.5], [0.6, 0.5], [0.7, 0.5], [0.8, 0.5], [0.9, 0.5],
			  #[1.0, 0.5], [0.0, 0.4], [0.1, 0.4], [0.2, 0.4], [0.3, 0.4],
			  #[0.4, 0.4], [0.5, 0.4], [0.6, 0.4], [0.7, 0.4], [0.8, 0.4],
			  #[0.9, 0.4], [1.0, 0.4], [0.0, 0.3], [0.1, 0.3], [0.2, 0.3],
			  #[0.3, 0.3], [0.4, 0.3], [0.5, 0.3], [0.6, 0.3], [0.7, 0.3],
			  #[0.8, 0.3], [0.9, 0.3], [1.0, 0.3], [0.0, 0.2], [0.1, 0.2],
			  #[0.2, 0.2], [0.3, 0.2], [0.4, 0.2], [0.5, 0.2], [0.6, 0.2],
			  #[0.7, 0.2], [0.8, 0.2], [0.9, 0.2], [1.0, 0.2], [0.0, 0.1],
			  #[0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.4, 0.1], [0.5, 0.1],
			  #[0.6, 0.1], [0.7, 0.1], [0.8, 0.1], [0.9, 0.1], [1.0, 0.1],
			  #[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0],
			  #[0.5, 0.0], [0.6, 0.0], [0.7, 0.0], [0.8, 0.0], [0.9, 0.0],
			  #[1.0, 0.0]])
#
## 0 : normal    1 : coleccionable
#
#y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			  #1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			  #1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
			  #0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
			  #0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
			  #0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
			  #0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
			  #0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
			  #0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
			  #0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
			  #0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
#
#func _ready() -> void:
## Para reproducibilidad
#np.random.seed(0)
#
#class RedNeuronal:
  #def __init__(self, x, y):
	## datos de entrenamiento
	#self.x = x
	## clase asociada a los datos de entrenamiento
	#self.y = y
	## estructura de la red e inicialización aleatoria de los pesos
	#self.pesos1 = np.random.rand(4)
	#self.sesgos1 = np.random.rand(2)
	#self.pesos2 = np.random.rand(2)
	#self.sesgos2 = np.random.rand(1)
#
  #def entrenamiento(self, tasa_aprendizaje=0.1, epocas=1000):
	## entrenamiento por k epocas
	#for k in range(epocas):
	  #error = 0
	  ## Para cada epoca k:
	  ## 1) haz propagación hacia adelante con cada instancia i
	  ## 2) calcula el error cuadrático y los gradientes
	  ## 3) actualiza los pesos
	  #for i in range(self.x.shape[0]):
		## Entradas a las neuronas sigmoides ocultas
		#suma_o1 = self.x[i][0]*self.pesos1[0] + self.x[i][1]*self.pesos1[2] + self.sesgos1[0]
		#suma_o2 = self.x[i][0]*self.pesos1[1] + self.x[i][1]*self.pesos1[3] + self.sesgos1[1]
		## Salidas de las neuronas sigmoides ocultas
		#salida_o1 =  1/(1 + np.exp(-suma_o1))
		#salida_o2 = 1/(1 + np.exp(-suma_o2))
		## Entrada de la neurona sigmoide de la capa de salida
		#suma_s = salida_o1*self.pesos2[0] + salida_o2*self.pesos2[1] + self.sesgos2[0]
		## Salida de la red neuronal
		#y_gorro = 1/(1 + np.exp(-suma_s))
#
		## Cálculo del error cuadrático
		#error += (1/2)*(self.y[i] - y_gorro)**2
#
		## Cálculo de los gradientes
		#gradiente_p21 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * salida_o1
		#gradiente_p22 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * salida_o2
		#gradiente_sesgo21 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * 1
#
		#gradiente_p11 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * \
						 #self.pesos2[0] * (salida_o1 * (1 - salida_o1)) * self.x[i][0]
		#gradiente_p13 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * \
						 #self.pesos2[0] * (salida_o1 * (1 - salida_o1)) * self.x[i][1]
		#gradiente_sesgo11 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * \
							 #self.pesos2[0] * (salida_o1 * (1 - salida_o1)) * 1
#
		#gradiente_p12 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * \
						 #self.pesos2[1] * (salida_o2 * (1 - salida_o2)) * self.x[i][0]
		#gradiente_p14 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * \
						 #self.pesos2[1] * (salida_o2 * (1 - salida_o2)) * self.x[i][1]
		#gradiente_sesgo12 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * \
							 #self.pesos2[1] * (salida_o2 * (1 - salida_o2)) * 1
#
		## Actualización de los pesos
		#self.pesos1[0] -= tasa_aprendizaje * gradiente_p11
		#self.pesos1[1] -= tasa_aprendizaje * gradiente_p12
		#self.pesos1[2] -= tasa_aprendizaje * gradiente_p13
		#self.pesos1[3] -= tasa_aprendizaje * gradiente_p14
		#self.sesgos1[0] -= tasa_aprendizaje * gradiente_sesgo11
		#self.sesgos1[1] -= tasa_aprendizaje * gradiente_sesgo12
		#self.pesos2[0] -= tasa_aprendizaje * gradiente_p21
		#self.pesos2[1] -= tasa_aprendizaje * gradiente_p22
		#self.sesgos2[0] -= tasa_aprendizaje * gradiente_sesgo21
	  #print(error)
#
  #def clasificacion(self, x1, x2):
	## Propagación hacia adelante con la nueva instancia (x1, x2) a evaluar
	#suma_o1 = x1*self.pesos1[0] + x2*self.pesos1[2] + self.sesgos1[0]
	#suma_o2 = x1*self.pesos1[1] + x2*self.pesos1[3] + self.sesgos1[1]
	#salida_o1 = 1/(1 + np.exp(-suma_o1))
	#salida_o2 = 1/(1 + np.exp(-suma_o2))
	#suma_s = salida_o1*self.pesos2[0] + salida_o2*self.pesos2[1] + self.sesgos2[0]
	#y_gorro = 1/(1 + np.exp(-suma_s))
	#return round(y_gorro)
#
## Crea una Red Neuronal Artificial
#red_neuronal = RedNeuronal(x, y)
#red_neuronal.entrenamiento()
#
	#prints( " hola neurona ")

extends Control

var x = [[0.0,0.0], [1.0,0.0], [0.0,1.0], [1.0,1.0]]
var y = [0, 1, 1, 0]


func _ready():
	randomize()
	#var x = [[0.0, 1.0], [0.1, 1.0], ...]  # Reemplaza con tu array de datos
	#var y = [0, 0, 0, ...]  # Reemplaza con tu array de etiquetas
	var n = randi()  % 4
	var red_neuronal = RedNeuronal.new(x, y)
	prints(red_neuronal.clasificacion(x[n][0], x[n][1]))
	red_neuronal.entrenamiento()
	print("Entrenamiento completado")
	
	
	for k in range(4):
	
		var forward = red_neuronal.clasificacion(x[k][0], x[k][1])
		var presicion = y[k]
		var redondear = round(forward )
		if redondear == presicion :
			prints(k,"la red acerto felicidades :)  redondeo ", redondear, " el dato :" ,y[k] , " la salida bruta : " ,forward)
		else:
			push_warning("la red erro prueba entrenar mas :)  redondeo ", redondear, " el dato :" ,y[k] , " la salida bruta : " ,forward)
			pass





class RedNeuronal:
	var x
	var y
	var pesos1 = []
	var sesgos1 = []
	var pesos2 = []
	var sesgos2 = []

	func _init(x, y):
		self.x = x
		self.y = y

		# Inicialización de los pesos y sesgos
		for i in range(6):
			self.pesos1.append(randf())
		for i in range(3):
			self.sesgos1.append(randf())
		for i in range(3):
			self.pesos2.append(randf())
		self.sesgos2.append(randf())

	func entrenamiento(tasa_aprendizaje = 0.4, epocas = 4000):
		for k in range(epocas ):
			var error = 0.0
			for i in range(self.x.size()):
				# Entradas a las neuronas sigmoides ocultas
				var suma_o1 = self.x[i][0] * self.pesos1[0] + self.x[i][1] * self.pesos1[2] + self.sesgos1[0]
				var suma_o2 = self.x[i][0] * self.pesos1[1] + self.x[i][1] * self.pesos1[3] + self.sesgos1[1]
				#var suma_o3 = 
				# Salidas de las neuronas sigmoides ocultas
				var salida_o1 = 1.0 / (1.0 + exp(-suma_o1))
				var salida_o2 = 1.0 / (1.0 + exp(-suma_o2))
				
				# Entrada de la neurona sigmoide de la capa de salida
				var suma_s = salida_o1 * self.pesos2[0] + salida_o2 * self.pesos2[1] + self.sesgos2[0]
				
				# Salida de la red neuronal
				var y_gorro = 1 / (1 + exp(-suma_s))

				# Cálculo del error cuadrático
				error += (0.5) * pow(self.y[i] - y_gorro, 2)
				
				# Cálculo de los gradientes
				var gradiente_p21 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * salida_o1
				var gradiente_p22 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * salida_o2
				var gradiente_sesgo21 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * 1.0

				var gradiente_p11 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * self.pesos2[0] * (salida_o1 * (1 - salida_o1)) * self.x[i][0]
				var gradiente_p13 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * self.pesos2[0] * (salida_o1 * (1 - salida_o1)) * self.x[i][1]
				var gradiente_sesgo11 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * self.pesos2[0] * (salida_o1 * (1 - salida_o1)) * 1

				var gradiente_p12 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * self.pesos2[1] * (salida_o2 * (1 - salida_o2)) * self.x[i][0]
				var gradiente_p14 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * self.pesos2[1] * (salida_o2 * (1 - salida_o2)) * self.x[i][1]
				var gradiente_sesgo12 = (y_gorro - self.y[i]) * (y_gorro * (1 - y_gorro)) * self.pesos2[1] * (salida_o2 * (1 - salida_o2)) * 1

				# Actualización de los pesos
				self.pesos1[0] -= tasa_aprendizaje * gradiente_p11
				self.pesos1[1] -= tasa_aprendizaje * gradiente_p12
				self.pesos1[2] -= tasa_aprendizaje * gradiente_p13
				self.pesos1[3] -= tasa_aprendizaje * gradiente_p14
				self.sesgos1[0] -= tasa_aprendizaje * gradiente_sesgo11
				self.sesgos1[1] -= tasa_aprendizaje * gradiente_sesgo12
				self.pesos2[0] -= tasa_aprendizaje * gradiente_p21
				self.pesos2[1] -= tasa_aprendizaje * gradiente_p22
				self.sesgos2[0] -= tasa_aprendizaje * gradiente_sesgo21
			
			#print(error)

	func clasificacion(x1, x2):
		var suma_o1 = x1 * self.pesos1[0] + x2 * self.pesos1[2] + self.sesgos1[0]
		var suma_o2 = x1 * self.pesos1[1] + x2 * self.pesos1[3] + self.sesgos1[1]
		var salida_o1 = 1.0 / (1.0 + exp(-suma_o1))
		var salida_o2 = 1.0 / (1.0 + exp(-suma_o2))
		var suma_s = salida_o1 * self.pesos2[0] + salida_o2 * self.pesos2[1] + self.sesgos2[0]
		var y_gorro = 1.0 / (1.0 + exp(-suma_s))
		return y_gorro#round(y_gorro)

extends Control


#
#class NeuralNetwork:
	#const LEARNING_RATE = 0.5
	#
	#var num_inputs: int
	#var hidden_layer: NeuronLayer
	#var output_layer: NeuronLayer
	#
	#func _init(num_inputs: int, num_hidden: int, num_outputs: int, hidden_layer_weights = null, hidden_layer_bias = null, output_layer_weights = null, output_layer_bias = null):
		#self.num_inputs = num_inputs
		#hidden_layer_bias = hidden_layer_bias if hidden_layer_bias else randf()
		#output_layer_bias = output_layer_bias if output_layer_bias else randf()
		#
		#self.hidden_layer = NeuronLayer.new(num_hidden, hidden_layer_bias)
		#self.output_layer = NeuronLayer.new(num_outputs, output_layer_bias)
		#
		#_init_weights(hidden_layer_weights, output_layer_weights)
	#
	#func _init_weights(hidden_layer_weights, output_layer_weights):
		#var weight_num = 0
		#
		## Initialize hidden layer weights
		#for h in range(hidden_layer.neurons.size()):
			#for _i in range(self.num_inputs):
				#if hidden_layer_weights == null:
					#hidden_layer.neurons[h].weights.append(randf())
				#else:
					#hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
				#weight_num += 1
		#
		## Initialize output layer weights
		#weight_num = 0
		#for o in range(output_layer.neurons.size()):
			#for _h in range(hidden_layer.neurons.size()):
				#if output_layer_weights == null:
					#output_layer.neurons[o].weights.append(randf())
				#else:
					#output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
				#weight_num += 1
	#
	#func feed_forward(inputs: Array) -> Array:
		#var hidden_outputs = hidden_layer.feed_forward(inputs)
		#return output_layer.feed_forward(hidden_outputs)
	#
	#func train(training_inputs: Array, training_outputs: Array):
		#feed_forward(training_inputs)
		#
		## 1. Output neuron deltas
		#var pd_errors_output = []
		#for o in range(output_layer.neurons.size()):
			#var neuron = output_layer.neurons[o]
			#pd_errors_output.append(neuron.calculate_pd_error_wrt_total_net_input(training_outputs[o]))
		#
		## 2. Hidden neuron deltas
		#var pd_errors_hidden = []
		#for h in range(hidden_layer.neurons.size()):
			#var error_wrt_output = 0.0
			#for o in range(output_layer.neurons.size()):
				#error_wrt_output += pd_errors_output[o] * output_layer.neurons[o].weights[h]
			#pd_errors_hidden.append(error_wrt_output * hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input())
		#
		## 3. Update weights in output layer
		#for o in range(output_layer.neurons.size()):
			#var neuron = output_layer.neurons[o]
			#for w_ho in range(neuron.weights.size()):
				#var pd_error_wrt_weight = pd_errors_output[o] * neuron.inputs[w_ho]
				#neuron.weights[w_ho] -= LEARNING_RATE * pd_error_wrt_weight
		#
		## 4. Update weights in hidden layer
		#for h in range(hidden_layer.neurons.size()):
			#var neuron = hidden_layer.neurons[h]
			#for w_ih in range(neuron.weights.size()):
				#var pd_error_wrt_weight = pd_errors_hidden[h] * neuron.inputs[w_ih]
				#neuron.weights[w_ih] -= LEARNING_RATE * pd_error_wrt_weight
	#
	#func calculate_total_error(training_sets: Array) -> float:
		#var total_error = 0.0
		#for set in training_sets:
			#var training_inputs = set[0]
			#var training_outputs = set[1]
			#feed_forward(training_inputs)
			#for o in range(output_layer.neurons.size()):
				#total_error += output_layer.neurons[o].calculate_error(training_outputs[o])
		#return total_error
#
#
#class NeuronLayer:
	#var neurons: Array
	#var bias: float
	#
	#func _init(num_neurons: int, bias: float):
		#self.bias = bias
		#for _i in range(num_neurons):
			#neurons.append(Neuron.new(self.bias))
	#
	#func feed_forward(inputs: Array) -> Array:
		#var outputs = []
		#for neuron in neurons:
			#outputs.append(neuron.calculate_output(inputs))
		#return outputs
#
#class Neuron:
	#var weights: Array
	#var inputs: Array
	#var output: float
	#var bias: float
	#
	#func _init(bias: float):
		#self.bias = bias
		#self.weights = []
	#
	#func calculate_output(inputs: Array) -> float:
		#self.inputs = inputs
		#self.output = _sigmoid(_calculate_total_net_input())
		#return self.output
	#
	#func _calculate_total_net_input() -> float:
		#var total = 0.0
		#for i in range(inputs.size()):
			#total += inputs[i] * weights[i]
		#return total + bias
	#
	#func _sigmoid(x: float) -> float:
		#return 1.0 / (1.0 + exp(-x))
	#
	#func _sigmoid_derivative(output: float) -> float:
		#return output * (1.0 - output)
#
	#func calculate_pd_error_wrt_total_net_input(target_output: float) -> float:
		#return calculate_pd_error_wrt_output(target_output) * _sigmoid_derivative(output)
	#
	#func calculate_error(target_output: float) -> float:
		#return 0.5 * pow(target_output - output, 2)
	#
	#func calculate_pd_error_wrt_output(target_output: float) -> float:
		#return -(target_output - output)
	#
	## Missing method: Add this function
	#func calculate_pd_total_net_input_wrt_input() -> float:
		#return _sigmoid_derivative(output)  # output * (1 - output)
#
#
## Example usage in Godot:
#
#func _ready():
	#var nn = NeuralNetwork.new(2, 2, 2, [0.15, 0.2, 0.25, 0.3], 0.35, [0.4, 0.45, 0.5, 0.55], 0.6)
	#for i in range(100):
		#nn.train([0.05, 0.1], [0.01, 0.99])
		#print("Iteration ", i, ", Total Error: ", nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]))
		#
## XOR example:
#
	#var training_sets = [
		#[[0, 0], [0]],
		#[[0, 1], [1]],
		#[[1, 0], [1]],
		#[[1, 1], [0]]
	#]
	#var nl = NeuralNetwork.new(2, 8, 1)
	#
	#for i in range(4000):
		#var training_data = training_sets.pick_random()
		#var inputs = training_data[0]
		#var targets = training_data[1]
		#nl.train(inputs, targets)
		#if i % 1000 == 0:
			#print("Iteration %d, Error: %.9f" % [i, nl.calculate_total_error(training_sets)])
	#
	## Test the neural network
	#print("Testing trained network:")
	#for set in training_sets:
		#var output = nl.feed_forward(set[0])
		#print("Input: %s, Predicted Output: %.5f, Target: %s" % [set[0], output[0], set[1]])
#

extends Control

#
#class NeuralNetwork:
	#var ni: int  # Número de nodos de entrada
	#var nh: int  # Número de nodos ocultos
	#var no: int  # Número de nodos de salida
	#
	#var ai: Array  # Activaciones de entrada
	#var ah: Array  # Activaciones ocultas
	#var ao: Array  # Activaciones de salida
	#
	#var wi: Array  # Pesos de entrada a capa oculta
	#var wo: Array  # Pesos de capa oculta a salida
	#
	#var ci: Array  # Cambios previos en los pesos de entrada
	#var co: Array  # Cambios previos en los pesos de salida
	#
	#func _init(num_inputs: int, num_hidden: int, num_outputs: int):
		## Configurar los nodos
		#ni = num_inputs + 1  # +1 por el nodo de sesgo
		#nh = num_hidden
		#no = num_outputs
		#
		## Inicializar activaciones
		#ai = []
		#ah = []
		#ao = []
		#for i in range(ni): ai.append(1.0)
		#for j in range(nh): ah.append(1.0)
		#for k in range(no): ao.append(1.0)
		#
		## Inicializar pesos con valores aleatorios
		#wi = _make_matrix(ni, nh, -0.2, 0.2)
		#wo = _make_matrix(nh, no, -2.0, 2.0)
		#
		## Inicializar cambios previos para momento
		#ci = _make_matrix(ni, nh, 0.0, 0.0)
		#co = _make_matrix(nh, no, 0.0, 0.0)
	#
	## Crear una matriz inicializada con valores aleatorios
	#func _make_matrix(rows: int, cols: int, a: float, b: float) -> Array:
		#var matrix = []
		#for i in range(rows):
			#var row = []
			#for j in range(cols):
				#row.append(randf_range(a, b))
			#matrix.append(row)
		#return matrix
	#
	## Función de activación Sigmoid
	#func sigmoid(x: float) -> float:
		#return tanh(x)
	#
	## Derivada de la función Sigmoid
	#func dsigmoid(y: float) -> float:
		#return 1.0 - pow(y, 2)
	#
	## Propagación hacia adelante
	#func update(inputs: Array) -> Array:
		#if inputs.size() != ni - 1:
			#print("Error: Número incorrecto de entradas")
			#return []
		#
		## Activaciones de entrada
		#for i in range(ni - 1):
			#ai[i] = inputs[i]
		#
		## Activaciones de la capa oculta
		#for j in range(nh):
			#var sum = 0.0
			#for i in range(ni):
				#sum += ai[i] * wi[i][j]
			#ah[j] = sigmoid(sum)
		#
		## Activaciones de la capa de salida
		#for k in range(no):
			#var sum = 0.0
			#for j in range(nh):
				#sum += ah[j] * wo[j][k]
			#ao[k] = sigmoid(sum)
		#
		#return ao.duplicate()
	#
	## Retropropagación
	#func back_propagate(targets: Array, learning_rate: float, momentum: float) -> float:
		#if targets.size() != no:
			#print("Error: Número incorrecto de valores objetivo")
			#return 0.0
		#
		## Cálculo de los errores de salida
		#var output_deltas = []
		#for k in range(no):
			#var error = targets[k] - ao[k]
			#output_deltas.append(dsigmoid(ao[k]) * error)
		#
		## Cálculo de los errores de la capa oculta
		#var hidden_deltas = []
		#for j in range(nh):
			#var error = 0.0
			#for k in range(no):
				#error += output_deltas[k] * wo[j][k]
			#hidden_deltas.append(dsigmoid(ah[j]) * error)
		#
		## Actualización de los pesos de salida
		#for j in range(nh):
			#for k in range(no):
				#var change = output_deltas[k] * ah[j]
				#wo[j][k] += learning_rate * change + momentum * co[j][k]
				#co[j][k] = change
		#
		## Actualización de los pesos de entrada
		#for i in range(ni):
			#for j in range(nh):
				#var change = hidden_deltas[j] * ai[i]
				#wi[i][j] += learning_rate * change + momentum * ci[i][j]
				#ci[i][j] = change
		#
		## Calcular el error total
		#var error = 0.0
		#for k in range(no):
			#error += 0.5 * pow(targets[k] - ao[k], 2)
		#return error
	#
	## Entrenar la red neuronal
	#func train(patterns: Array, iterations: int = 1000, learning_rate: float = 0.5, momentum: float = 0.1):
		#for i in range(iterations):
			#var error = 0.0
			#for p in patterns:
				#var inputs = p[0]
				#var targets = p[1]
				#update(inputs)
				#error += back_propagate(targets, learning_rate, momentum)
			#if i % 100 == 0:
				#print("Iteración %d, Error: %.5f" % [i, error])
	#
	## Probar la red
	#func test(patterns: Array):
		#for p in patterns:
			#var output = update(p[0])
			#print("Entrada: %s -> Salida: %.5f, Objetivo: %s" % [p[0], output[0], p[1]])
#
## Ejemplo de uso: función XOR
#func _ready():
	#var patterns = [
		#[[0, 0], [0]],
		#[[0, 1], [1]],
		#[[1, 0], [1]],
		#[[1, 1], [0]]
	#]
	#
	#var nn = NeuralNetwork.new(2, 2, 1)
	#nn.train(patterns, 1000, 0.5, 0.1)
	#print("Prueba de la red neuronal entrenada:")
	#nn.test(patterns)



extends Control
'''
################esta en desarollo aun falta mucho pero vamos ##############

Etiqueta Descripción
0	Camiseta/top
1	Pantalón
2	Jersey
3	Vestido
4	Abrigo
5	Sandalia
6	Camisa
7	Zapatillas de deporte
8	Bolsa
9	Botín
'''
# Diccionario de etiquetas con su descripción
var etiquetas = {
	0: "Camiseta/top ",
	1: "Pantalón 👖",
	2: "Jersey",
	3: "Vestido 👗",
	4: "Abrigo 👕",
	5: "Sandalia",
	6: "Camisa 🧥",
	7: "Zapatillas de deporte 👟",
	8: "Bolsa 👜",
	9: "Botín 👢"
}

var inig = NeuralNetwork
var history = ""
var load_data
var image_paths = []
var tag = []
var tag_data = []
var dir_fashion = "C:/Users/Emabe/Desktop/entrena/fashion/train/"
var max = 100# para todo cargar imagen leer el cvs y mas 
var leer = max
var real_data
var fashion = "C:/Users/Emabe/Desktop/entrena/fashion/train.csv"

func _ready():
	''' solo ia '''
	randomize()
	var discriminator_structure = [784,1280,512,320,80,10]  # estructura de ejemplo 
	var use_bias = true
	
	#var gan = IMG.new(discriminator_structure, use_bias)
	#gan.load_gan("res://data/etiqueta_img_train.bin")

	
	
	
	
	''' archivo csv'''
	
	
	var file_path = fashion # Ruta al archivo CSV
	var file = FileAccess.open(file_path, FileAccess.READ)




	if file:
		while not file.eof_reached() :
			
			if leer <= 0:
				break
			var line = file.get_line()
			var columns = line.split(",")
			var first_value = columns[0]  # primer valor de la línea
			var last_value = columns[columns.size() - 1]  # ultimo valor de la línea
			print("primer valor: ", first_value.to_int(), ", ultimo valor: ", last_value.to_int())
			tag.append(last_value.to_int()) # genero el tag en el index de la imagen
			leer -= 1
		
		file.close()
	else:
		print("el archivo no existe: ", file_path)
		
	
	'''etiquetamos '''
	for i in range(tag.size()):
		#var etiqueta = i 
		var array = etiqueta(tag[i])
		tag_data.append(array)
		print("Etiqueta: ", etiquetas[tag[i]], " Array: ", array)
	
	
	''' cargamos los archivos '''
	
	
	
	
	
	var size = 28  # Tamaño de las imagenes
	'''dir no lee en orden '''
	#dir_contents(dir_fashion , max) # cargamos los archivos primero , cargamos todo ?? 
	dir_count(dir_fashion, max)
	
	while !load_data :
		load_data = load_training_images(image_paths, size)

	print("Datos cargados: ", load_data.size())
	prints("tag cargados: " ,tag_data.size())
	prints(tag_data)
	
	inig = NeuralNetwork.new("Sigmoid", 0.33, false)
	inig.add_layer(784,512) # Example: input layer with 2 inputs, 3 neurons
	inig.add_layer(512, 64)
	#nn.add_layer(20, 20)
	inig.add_layer(64, 10)
	#nn.add_layer(3, 3)
	#inig.add_layer(64, 10) # Hidden layer to output with 1 neuron
	#inig.set_loss_function(BNNET.LossFunctions.MAE)
	#inig.activation_to_string(BNNET.ActivationFunctions.softmax)
	#inig.use_Adam(0.01)
	#inig.use_Adadelta(0.5)
	#inig.set_loss_function(BNNET.LossFunctions.CCE)
	#inig.use_Rprop(0.9)
	#inig.set_batch_size(1)
	#inig.use_NAG(0.9,0.1)
	#inig.use_Yogi(0.1)
	
	
	''' entrenamiento '''
	for li in range(20):
		for i in range(10):
			#inig.load_data("res://data/etiqueta_img_train.bin")
			trai(1,inig,load_data,tag_data,i)

		
			#inig.save_binary("res://data/etiqueta_img_train.bin")
	prints("guardamos")
	
	
	
	#loss_tri(inig,load_data , tag_data)

	

''' solo cargamos y comprobamos '''











func trai(bucle ,inig,load_data , tag_data , idex):

	var count_index =  0
	
	
	for i in range(bucle):
		

		
		prints("entrenamiento en ronda 10 por : " , i , " de tantas : " , bucle )
		for j in range(1):
			for k in range(load_data.size()):
				var load_data2 = []
				var tag_data2 = []
				var array = etiqueta(idex)
				if tag_data[k] == array and array != null:
					count_index += 1
					load_data2.append(load_data[k])
					tag_data2.append(tag_data[k])
					prints(k)
				else:
					continue
			#inig.apply_gradients(0.01)
				for h in range(1):
					#inig.set_input( load_data2[0])
					#inig.set_target( tag_data2[0])
					#inig.propagate_forward()
					var train_new = []
					var data_pass = []
					#inig.apply_gradients(0.01)
					train_new.append(load_data2[0])
					train_new.append(tag_data2[0])
					data_pass.append(train_new)
					#prints(train_new)
					#inig.propagate_backward()
					inig.train(data_pass,1000)
					#inig.propagate_backward()
					if j % 2 == 0:
						loss_tri(inig,data_pass)
						print("✨✨✨entrenando la red idex -:",idex," Iteración✨✨: ", j ," %  ✨✨porciento✨✨ de: ", k , " :✨✨ bucles✨✨")
	prints("total index " , count_index ," en modo :" ,idex )


func loss_tri(nn,train):
	nn.test(train)

		#
		#print("Loss: ", nn.get_loss(load_data2, tag_data2) , "  imagen: " , hit,".png")
		#
		#
		#var array = etiqueta(tag[hit])
		#
		#print("😊Etiqueta😊: ", etiquetas[tag[hit]], "🌟 Array 🌟: ", array)
		#prints("🚀data out clasifi🚀: ",nn.get_output()," ⭐️data⭐️ :", tag_data[0])
		#prints("⭐️num aleatorio⭐️: " ,hit )
	pass







func load_training_images(image_paths: Array, size: int) -> Array:
	var loaded_data = []
	
	for path in image_paths:
		var image = Image.new()
		var error = image.load(path)
		
		if error != OK:
			print("Failed to load image: ", path)
			continue
		
		if image.get_width() != size or image.get_height() != size:
			print("Image size does not match. Width: ", image.get_width(), " Height: ", image.get_height(), " Expected: ", size)
			continue
		
		var image_data = []
		for y in range(size):
			for x in range(size):
				var color = image.get_pixel(x, y)
				var value = color.r  # Asumiendo escala de grises, usar solo el canal rojo
				image_data.append(value)
		
		loaded_data.append(image_data)
	
	return loaded_data


func dir_count(path, count):
	for i in range(count):
		var file_name = str(i) + ".png"
		prints(file_name)
		image_paths.append(path + "/" + file_name)
	

func dir_contents(path , max):
	
	var dir = DirAccess.open(path)
	if dir:
		dir.list_dir_begin()
		var file_name = dir.get_next()
		while file_name != "" and max > 0:
			max -= 1
			if dir.current_is_dir():
				print("⭐️DIRECTORIO ENCONTRADO⭐️ : " + file_name)
			else:
				print("⭐ARCHIVO ENCONTRADO⭐️ : " + file_name)
				print("f⭐️EXTENCION DEL ARCHIVO⭐️: " + file_name.get_extension())

				if file_name.get_extension() == "png" or file_name.get_extension() == "jpg":
					prints("estencion gd")
					image_paths.append(path + "/" + file_name)
					prints( "⭐️DIRECTORIO⭐️  ",path + "/" + file_name)
			file_name = dir.get_next()
	else:
		print("⭐️ EDITOR FALLO : An error occurred when trying to access the path ⭐️")






func etiqueta(etiqueta: int) -> Array:
	var array_resultado = Array()
	for i in range(10):
		if i == etiqueta:
			array_resultado.append(1)
		else:
			array_resultado.append(0)
	return array_resultado














# NeuralNetwork Class
class NeuralNetwork:
	var activ_func
	var layers = []
	var learning_rate
	var debug
	
	func _init(activation_func = "Sigmoid", learning_rate = 0.01, debug = true):
		self.activ_func = ActivationFunction.new(activation_func)
		self.learning_rate = learning_rate
		self.debug = debug
	
	func add_layer(n_inputs: int, n_neurons: int):
		var layer = NeuralLayer.new(n_inputs, n_neurons, self.activ_func)
		layers.append(layer)
	
	func feed_forward(inputs: Array) -> Array:
		for i in range(len(layers)):
			inputs = layers[i].feed_forward(inputs)
			if debug:
				print("Layer %d, Output: %s" % [i+1, inputs])
		return inputs
	
	func feed_backward(targets: Array):
		if len(targets) != len(layers[-1].neurons):
			push_error("Wrong target numbers")
		
		# Calculate deltas for output layer
		for j in range(len(layers[-1].neurons)):
			var error = -(targets[j] - layers[-1].neurons[j].output)
			layers[-1].neurons[j].calculate_delta(error)
		
		if debug:
			print("Output Layer Deltas: ", layers[-1].get_deltas())
		
		# Backpropagation for hidden layers
		for l in range(len(layers) - 2, -1, -1):
			var curr_layer = layers[l]
			var next_layer = layers[l + 1]
			for i in range(len(curr_layer.neurons)):
				var total_error = 0.0
				for j in range(len(next_layer.neurons)):
					total_error += next_layer.neurons[j].delta * next_layer.neurons[j].weights[i]
				curr_layer.neurons[i].calculate_delta(total_error)
			if debug:
				print("Layer %d Deltas: %s" % [l+1, curr_layer.get_deltas()])
	
	func update_weights():
		for layer in layers:
			layer.update_weights(learning_rate)
	
	func calculate_total_error(dataset: Array) -> float:
		var total_error = 0.0
		for data in dataset:
			var inputs = data[0]
			var targets = data[1]
			var outputs = feed_forward(inputs)
			for i in range(len(targets)):
				total_error += pow(targets[i] - outputs[i], 2)
		return total_error / len(dataset)
	
	func train(dataset: Array, iterations: int = 100):
		print("\n> Training...")
		for i in range(iterations):
			for data in dataset:
				var inputs = data[0]
				var targets = data[1]
				feed_forward(inputs)
				feed_backward(targets)
				update_weights()
			var error = calculate_total_error(dataset)
			#print("Iteration %d, Error: %f" % [i+1, error])
		print("> Training Complete!")
	
		
		
	func test(dataset : Array) -> void:
		var tag_2 = 0
		print('\n> Testing...')
		print("------------------------------")  # Puedes reemplazar esto con una constante STR_REPORT_BROADER si lo deseas

	# Iterar sobre cada dato de entrada en el dataset
		for j in range(len(dataset)):
			var inputs = dataset[j][0]
			var targets = dataset[j][1]
			if debug:
				print('\n>>> data #{}'.format(j + 1))
		
		# Realizar la propagación hacia adelante para obtener los resultados
			var actual_outputs = feed_forward(inputs)
			
			if round(actual_outputs[0]) == targets[0]:
				prints("acerto")
			else:
				push_error("error la red es tonta ")
				tag_2 += 1
			print(j, inputs, actual_outputs, targets)
	
	# Calcular el error total
		var total_error = calculate_total_error(dataset)

		print("------------------------------")  # Puedes reemplazar esto con una constante STR_REPORT_BROADER si lo deseas
		print('Testing Finish. Error: ',(total_error))  # Muestra el error total
		prints("error tantas :", tag_2)


# NeuralLayer Class
class NeuralLayer:
	var neurons = []
	
	func _init(n_inputs: int, n_neurons: int, activ_func):
		for _i in range(n_neurons):
			neurons.append(Neuron.new(n_inputs, activ_func))
	
	func feed_forward(inputs: Array) -> Array:
		var outputs = []
		for neuron in neurons:
			outputs.append(neuron.calculate_output(inputs))
		return outputs
	
	func update_weights(learning_rate: float):
		for neuron in neurons:
			neuron.update_weights(learning_rate)
	
	func get_deltas():
		var deltas = []
		for neuron in neurons:
			deltas.append(neuron.delta)
		return deltas




class Neuron:

	var weights = []
	var bias = 1.0
	var output = 0.0
	var delta = 0.0
	var inputs = []
	var activation_function : ActivationFunction

	func _init(n_weights: int, activ_func: ActivationFunction) -> void:
		self.weights = []
		for _i in range(n_weights):
			weights.append(randf()) # Inicializa pesos aleatorios
		self.activation_function = activ_func

	func calculate_output(inputs: Array) -> float:
		self.inputs = inputs
		var total = bias
		for i in range(len(inputs)):
			total += inputs[i] * weights[i]
		output = activation_function.activation_func.call(total)  # Llama correctamente a la función de activación
		return output

	func calculate_delta(error: float) -> void:
		delta = error * activation_function.derivative_func.call(output)  # Llama correctamente a la derivada

	func update_weights(learning_rate: float) -> void:
		for i in range(len(weights)):
			weights[i] -= learning_rate * delta * inputs[i]
		bias -= learning_rate * delta



#
func _ready3() -> void:
	
	
	var data_new = []
	prints("funca test 3")
	var nn = NeuralNetwork.new("Sigmoid", 0.3, false)
	nn.add_layer(2, 2) # Example: input layer with 2 inputs, 3 neurons
	#nn.add_layer(2, 3)
	#nn.add_layer(20, 20)
	#nn.add_layer(3, 4)
	#nn.add_layer(3, 3)
	nn.add_layer(2, 1) # Hidden layer to output with 1 neuron
	var x : Array = [[0.0, 1.0], [0.1, 1.0], [0.2, 1.0], [0.3, 1.0], [0.4, 1.0],
	[0.5, 1.0], [0.6, 1.0], [0.7, 1.0], [0.8, 1.0], [0.9, 1.0],
	[1.0, 1.0], [0.0, 0.9], [0.1, 0.9], [0.2, 0.9], [0.3, 0.9],
	[0.4, 0.9], [0.5, 0.9], [0.6, 0.9], [0.7, 0.9], [0.8, 0.9],
	[0.9, 0.9], [1.0, 0.9], [0.0, 0.8], [0.1, 0.8], [0.2, 0.8],
	[0.3, 0.8], [0.4, 0.8], [0.5, 0.8], [0.6, 0.8], [0.7, 0.8],
	[0.8, 0.8], [0.9, 0.8], [1.0, 0.8], [0.0, 0.7], [0.1, 0.7],
	[0.2, 0.7], [0.3, 0.7], [0.4, 0.7], [0.5, 0.7], [0.6, 0.7],
	[0.7, 0.7], [0.8, 0.7], [0.9, 0.7], [1.0, 0.7], [0.0, 0.6],
	[0.1, 0.6], [0.2, 0.6], [0.3, 0.6], [0.4, 0.6], [0.5, 0.6],
	[0.6, 0.6], [0.7, 0.6], [0.8, 0.6], [0.9, 0.6], [1.0, 0.6],
	[0.0, 0.5], [0.1, 0.5], [0.2, 0.5], [0.3, 0.5], [0.4, 0.5],
	[0.5, 0.5], [0.6, 0.5], [0.7, 0.5], [0.8, 0.5], [0.9, 0.5],
	[1.0, 0.5], [0.0, 0.4], [0.1, 0.4], [0.2, 0.4], [0.3, 0.4],
	[0.4, 0.4], [0.5, 0.4], [0.6, 0.4], [0.7, 0.4], [0.8, 0.4],
	[0.9, 0.4], [1.0, 0.4], [0.0, 0.3], [0.1, 0.3], [0.2, 0.3],
	[0.3, 0.3], [0.4, 0.3], [0.5, 0.3], [0.6, 0.3], [0.7, 0.3],
	[0.8, 0.3], [0.9, 0.3], [1.0, 0.3], [0.0, 0.2], [0.1, 0.2],
	[0.2, 0.2], [0.3, 0.2], [0.4, 0.2], [0.5, 0.2], [0.6, 0.2],
	[0.7, 0.2], [0.8, 0.2], [0.9, 0.2], [1.0, 0.2], [0.0, 0.1],
	[0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.4, 0.1], [0.5, 0.1],
	[0.6, 0.1], [0.7, 0.1], [0.8, 0.1], [0.9, 0.1], [1.0, 0.1],
	[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0],
	[0.5, 0.0], [0.6, 0.0], [0.7, 0.0], [0.8, 0.0], [0.9, 0.0],
	[1.0, 0.0]]

# 0 : normal    1 : coleccionable

	var y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
	0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
	0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
	0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
	0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
	0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
	0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
	0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
 #Los datos x y y no se incluyen aquí porque ya los tienes definidos.
	for i in range(x.size()):
		data_new.append([x[i] , [y[i]]])
	var dataset = [
		[[0, 0], [0]],
		[[0, 1], [1]],
		[[1, 0], [1]],
		[[1, 1], [0]],
		[[0, 0], [0]],
		[[0, 1], [1]],
		[[1, 0], [1]],
		[[1, 1], [0]],
]
	nn.train(data_new, 900)
	nn.test(data_new)


class_name Calendar

enum Month { JAN = 1, FEB = 2, MAR = 3, APR = 4, MAY = 5, JUN = 6, JUL = 7,
		AUG = 8, SEP = 9, OCT = 10, NOV = 11, DEC = 12 }

const MONTH_NAME = [ 
		"Jan", "Feb", "Mar", "Apr", 
		"May", "Jun", "Jul", "Aug", 
		"Sep", "Oct", "Nov", "Dec" ]

const WEEKDAY_NAME = [ 
		"Sunday", "Monday", "Tuesday", "Wednesday", 
		"Thursday", "Friday", "Saturday" ]

func get_days_in_month(month : int, year : int) -> int:
	var number_of_days : int
	if(month == Month.APR || month == Month.JUN || month == Month.SEP
			|| month == Month.NOV):
		number_of_days = 30
	elif(month == Month.FEB):
		var is_leap_year = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
		if(is_leap_year):
			number_of_days = 29
		else:
			number_of_days = 28
	else:
		number_of_days = 31
	
	return number_of_days

func get_weekday(day : int, month : int, year : int) -> int:
	var t : Array = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
	if(month < 3):
		year -= 1
	return (year + year/4 - year/100 + year/400 + t[month - 1] + day) % 7

func get_weekday_name(day : int, month : int, year : int) -> String:
	var day_num = get_weekday(day, month, year)
	return WEEKDAY_NAME[day_num]

func get_month_name(month : int) -> String:
	return MONTH_NAME[month - 1]

func hour() -> int:
	return Time.get_datetime_dict_from_system()["hour"]

func minute() -> int:
	return  Time.get_datetime_dict_from_system()["minute"]

func second() -> int:
	return  Time.get_datetime_dict_from_system()["second"]

func day() -> int:
	return  Time.get_datetime_dict_from_system()["day"]

func weekday() -> int:
	return  Time.get_datetime_dict_from_system()["weekday"]

func month() -> int:
	return  Time.get_datetime_dict_from_system()["month"]

func year() -> int:
	return  Time.get_datetime_dict_from_system()["year"]

func daylight_savings_time() -> int:
	return dst()

func dst() -> int:
	return  Time.get_datetime_dict_from_system()["dst"]


class_name Date

var day:int:
	set(v):
		day = v
	get:
		return day
var month:int:
	set(v):
		month = v
	get:
		return month
var year : int:
	set(v):
		year = v
	get:
		return year

func _init(day : int = Time.get_datetime_dict_from_system()["day"], 
		month : int = Time.get_datetime_dict_from_system()["month"], 
		year : int = Time.get_datetime_dict_from_system()["year"]):
	self.day = day
	self.month = month
	self.year = year

# Supported Date Formats:
# DD : Two digit day of month
# MM : Two digit month
# YY : Two digit year
# YYYY : Four digit year
func date(date_format = "DD-MM-YY") -> String:
	if("DD".is_subsequence_of(date_format)):
		date_format = date_format.replace("DD", "%02d" % get_day())
	if("MM".is_subsequence_of(date_format)):
		date_format = date_format.replace("MM", "%02d" % get_month())
	if("YYYY".is_subsequence_of(date_format)):
		date_format = date_format.replace("YYYY","%04d" % get_year())
	elif("YY".is_subsequence_of(date_format)):
		date_format = date_format.replace("YY", str(get_year()).substr(2))
	return date_format

func get_day() -> int:
	return day

func get_month() -> int:
	return month

func get_year() -> int:
	return year

func set_day(_day : int):
	day = _day

func set_month(_month : int):
	month = _month

func set_year(_year : int):
	year = _year

func change_to_prev_month():
	var selected_month = get_month()
	selected_month -= 1
	if(selected_month < 1):
		set_month(12)
		set_year(get_year() - 1)
	else:
		set_month(selected_month)

func change_to_next_month():
	var selected_month = get_month()
	selected_month += 1
	if(selected_month > 12):
		set_month(1)
		set_year(get_year() + 1)
	else:
		set_month(selected_month)

func change_to_prev_year():
	set_year(get_year() - 1)

func change_to_next_year():
	set_year(get_year() + 1)



extends Control




var key = "Este es un ejemp" # Key must be either 16 or 32 bytes.
var data = "Este es un ejemplo de un mnsaje muy largo tanto que no me preocupo
  por qu va hacer rellenado con mas texto ,como dice en la documentaion de godot " # Data size must be multiple of 16 bytes, apply padding if needed.
#var aes = AESContext.new()



func _ready():
	prints( " prueba de classe eb aes contexto ")
	prints( " prueba de classe eb aes contexto ")
	prints( " prueba de classe eb aes contexto ")
	prints( " prueba de classe eb aes contexto ")
	
	var largo = data.length()
	var prueba = aes_tool.new()
	
	#region > encripto de prueba a ver si sale bien '''
	
	var encrip = prueba.encrypt_aes_ecb(key, data)
	prints(encrip)
	
	''' desencripto a ver que pasa '''
	
	var derip = prueba.decrypt_aes_ecb(key, encrip)
	
	
	
	prints(derip.get_string_from_utf8())
	
	'''veo el string y lo corto si es mas largo del original'''
	
	prints(prueba.string_to_length(derip.get_string_from_utf8(), largo))
	
	'''uso una funcion que retorna el string '''
	
	prints(prueba.string_to_aes_ecb(key,encrip,largo))
	
#endregion
	
	pass # Replace with function body.


extends Node

class_name aes_tool

var aes = AESContext.new()

#region : test
'''retorna si es un multiplo de 16'''
#endregion


func is_multiple_of_16(length: int) -> bool:
	return length % 16 == 0

#region
''' retorna siempre un multiplo de 16 al string'''
#endregion


func string_to_multiple_of_16(input_string: String) -> String:
	var length := input_string.length()
	if is_multiple_of_16(length):
		return input_string
		
	prints(" no es multiplo de 16 ")
	
	
	var characters := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
	var random := RandomNumberGenerator.new()
	random.randomize()
	
	while !is_multiple_of_16(length):
		input_string += characters[random.randi_range(0, characters.length() - 1)]
		length = input_string.length()
	
	return input_string



#region
''' cortamos un string para un largo expeifico'''
#endregion


func string_to_length(input_string: String, max_length: int) -> String:
	if input_string.length() > max_length:
		return input_string.substr(0, max_length)
	return input_string


#region
'''hacemos el encriptado y retorna u packetbytearray'''
#endregion



func encrypt_aes_ecb(key: String, text: String) -> PackedByteArray:
	''' relleno del string '''
	var texto = string_to_multiple_of_16(text)
	
	aes.start(AESContext.MODE_ECB_ENCRYPT, key.to_utf8_buffer()) #AES en modo ECB (Electronic Codebook)
	var encrypted = aes.update(texto.to_utf8_buffer())
	#prints(encrypted , "  encripted data ecb")
	aes.finish()
	
	
	return encrypted
	pass

#region
''' desenripta el paquetbytearray y devuelve un packetbytearray'''
#endregion


func decrypt_aes_ecb(key: String, encrypted: PackedByteArray) -> PackedByteArray:
	aes.start(AESContext.MODE_ECB_DECRYPT, key.to_utf8_buffer())
	var decrypted = aes.update(encrypted)
	#prints(decrypted.get_string_from_utf8() , "   decripted data ecb")
	aes.finish()

	return decrypted

func string_to_aes_ecb(key: String, encrypted: PackedByteArray , length: int) -> String:
	aes.start(AESContext.MODE_ECB_DECRYPT, key.to_utf8_buffer())
	var decrypted = aes.update(encrypted)
	var texto: String = string_to_length(decrypted.get_string_from_utf8() ,length)
	#texto = string_to_length(decrypted.get_string_from_utf8() ,length)
	aes.finish()

	return texto


#
#func _ready() -> void:
	#var original_string := "Hola, soy una cadena de prueba"
	#var padded_string := string_to_multiple_of_16(original_string)
	#print("Cadena original: ", original_string)
	#print("Cadena origina tamañol: ", original_string.length())
	#print("Cadena rellenada: ", padded_string)
	#print("Longitud de la cadena rellenada: ", padded_string.length())
	#
	#
	#
	#original_string = "Este es un ejemplo de una cadena larga que necesita ser recortada."
	#var max_length := 16
	#var trimmed_string := string_to_length(original_string, max_length)
	#
	#print("Cadena original: ", original_string)
	#print("Cadena recortada: ", trimmed_string)
	#print("Longitud de la cadena recortada: ", trimmed_string.length())



# cmake arguments
# CMAKE_BUILD_TYPE:			Compilation target (Debug or Release defaults to Debug)
#
# godot-cpp cmake arguments
# GODOT_GDEXTENSION_DIR:		Path to the directory containing GDExtension interface header and API JSON file
# GODOT_CPP_SYSTEM_HEADERS		Mark the header files as SYSTEM. This may be useful to suppress warnings in projects including this one.
# GODOT_CPP_WARNING_AS_ERROR	Treat any warnings as errors
# GODOT_ENABLE_HOT_RELOAD       Build with hot reload support. Defaults to YES for Debug-builds and NO for Release-builds.
# GODOT_CUSTOM_API_FILE:		Path to a custom GDExtension API JSON file (takes precedence over `gdextension_dir`)
# FLOAT_PRECISION:				Floating-point precision level ("single", "double")
#
# Android cmake arguments
# CMAKE_TOOLCHAIN_FILE:		The path to the android cmake toolchain ($ANDROID_NDK/build/cmake/android.toolchain.cmake)
# ANDROID_NDK:				The path to the android ndk root folder
# ANDROID_TOOLCHAIN_NAME:	The android toolchain (arm-linux-androideabi-4.9 or aarch64-linux-android-4.9 or x86-4.9 or x86_64-4.9)
# ANDROID_PLATFORM:			The android platform version (android-23)
# More info here: https://godot.readthedocs.io/en/latest/development/compiling/compiling_for_android.html
#
# Examples
#
# Builds a debug version:
# cmake .
# cmake --build .
#
# Builds a release version with clang
# CC=/usr/bin/clang CXX=/usr/bin/clang++ cmake -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" .
# cmake --build .
#
# Builds an android armeabi-v7a debug version:
# cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_NDK=$ANDROID_NDK \
#		-DANDROID_TOOLCHAIN_NAME=arm-linux-androideabi-4.9 -DANDROID_PLATFORM=android-23 -DCMAKE_BUILD_TYPE=Debug .
# cmake --build .
#
# Protip
# Generate the buildfiles in a sub directory to not clutter the root directory with build files:
# mkdir build && cd build && cmake -G "Unix Makefiles" .. && cmake --build .
#
# Todo
# Test build for Windows, Mac and mingw.

cmake_minimum_required(VERSION 3.13)
project(godot-cpp LANGUAGES CXX)

option(GENERATE_TEMPLATE_GET_NODE "Generate a template version of the Node class's get_node." ON)
option(GODOT_CPP_SYSTEM_HEADERS "Expose headers as SYSTEM." ON)
option(GODOT_CPP_WARNING_AS_ERROR "Treat warnings as errors" OFF)

# Add path to modules
list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/" )

# Set some helper variables for readability
set( compiler_is_clang "$<OR:$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:Clang>>" )
set( compiler_is_gnu "$<CXX_COMPILER_ID:GNU>" )
set( compiler_is_msvc "$<CXX_COMPILER_ID:MSVC>" )

# Default build type is Debug in the SConstruct
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
	set(CMAKE_BUILD_TYPE Debug)
endif()

# Hot reload is enabled by default in Debug-builds
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    option(GODOT_ENABLE_HOT_RELOAD "Build with hot reload support" ON)
else()
    option(GODOT_ENABLE_HOT_RELOAD "Build with hot reload support" OFF)
endif()

if(NOT DEFINED BITS)
	set(BITS 32)
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
		set(BITS 64)
	endif(CMAKE_SIZEOF_VOID_P EQUAL 8)
endif()

# Input from user for GDExtension interface header and the API JSON file
set(GODOT_GDEXTENSION_DIR "gdextension" CACHE STRING "")
set(GODOT_CUSTOM_API_FILE "" CACHE STRING "")

set(GODOT_GDEXTENSION_API_FILE "${GODOT_GDEXTENSION_DIR}/extension_api.json")
if (NOT "${GODOT_CUSTOM_API_FILE}" STREQUAL "")  # User-defined override.
	set(GODOT_GDEXTENSION_API_FILE "${GODOT_CUSTOM_API_FILE}")
endif()

set(FLOAT_PRECISION "single" CACHE STRING "")
if ("${FLOAT_PRECISION}" STREQUAL "double")
	add_definitions(-DREAL_T_IS_DOUBLE)
endif()

set(GODOT_COMPILE_FLAGS )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	# using Visual Studio C++
	set(GODOT_COMPILE_FLAGS "/utf-8") # /GF /MP

	if(CMAKE_BUILD_TYPE MATCHES Debug)
		set(GODOT_COMPILE_FLAGS "${GODOT_COMPILE_FLAGS} /MDd") # /Od /RTC1 /Zi
	else()
		set(GODOT_COMPILE_FLAGS "${GODOT_COMPILE_FLAGS} /MD /O2") # /Oy /GL /Gy
		STRING(REGEX REPLACE "/RTC(su|[1su])" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
		string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
	endif(CMAKE_BUILD_TYPE MATCHES Debug)

	add_definitions(-DNOMINMAX)
else()  # GCC/Clang
	if(CMAKE_BUILD_TYPE MATCHES Debug)
		set(GODOT_COMPILE_FLAGS "${GODOT_COMPILE_FLAGS} -fno-omit-frame-pointer -O0 -g")
	else()
		set(GODOT_COMPILE_FLAGS "${GODOT_COMPILE_FLAGS} -O3")
	endif(CMAKE_BUILD_TYPE MATCHES Debug)
endif()

# Disable exception handling. Godot doesn't use exceptions anywhere, and this
# saves around 20% of binary size and very significant build time (GH-80513).
option(GODOT_DISABLE_EXCEPTIONS ON "Force disabling exception handling code")
if (GODOT_DISABLE_EXCEPTIONS)
	if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
		set(GODOT_COMPILE_FLAGS "${GODOT_COMPILE_FLAGS} -D_HAS_EXCEPTIONS=0")
	else()
		set(GODOT_COMPILE_FLAGS "${GODOT_COMPILE_FLAGS} -fno-exceptions")
	endif()
else()
	if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
		set(GODOT_COMPILE_FLAGS "${GODOT_COMPILE_FLAGS} /EHsc")
	endif()
endif()

if (GODOT_ENABLE_HOT_RELOAD)
    set(GODOT_COMPILE_FLAGS "${GODOT_COMPILE_FLAGS} -D HOT_RELOAD_ENABLED")
endif()

# Generate source from the bindings file
find_package(Python3 3.4 REQUIRED) # pathlib should be present
if(GENERATE_TEMPLATE_GET_NODE)
	set(GENERATE_BINDING_PARAMETERS "True")
else()
	set(GENERATE_BINDING_PARAMETERS "False")
endif()

execute_process(COMMAND "${Python3_EXECUTABLE}" "-c" "import binding_generator; binding_generator.print_file_list(\"${GODOT_GDEXTENSION_API_FILE}\", \"${CMAKE_CURRENT_BINARY_DIR}\", headers=True, sources=True)"
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	OUTPUT_VARIABLE GENERATED_FILES_LIST
	OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_custom_command(OUTPUT ${GENERATED_FILES_LIST}
		COMMAND "${Python3_EXECUTABLE}" "-c" "import binding_generator; binding_generator.generate_bindings(\"${GODOT_GDEXTENSION_API_FILE}\", \"${GENERATE_BINDING_PARAMETERS}\", \"${BITS}\", \"${FLOAT_PRECISION}\", \"${CMAKE_CURRENT_BINARY_DIR}\")"
		VERBATIM
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		MAIN_DEPENDENCY ${GODOT_GDEXTENSION_API_FILE}
		DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/binding_generator.py
		COMMENT "Generating bindings"
)

# Get Sources
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.c**)
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS include/*.h**)

# Define our godot-cpp library
add_library(${PROJECT_NAME} STATIC
		${SOURCES}
		${HEADERS}
		${GENERATED_FILES_LIST}
)
add_library(godot::cpp ALIAS ${PROJECT_NAME})

include(GodotCompilerWarnings)

target_compile_features(${PROJECT_NAME}
	PRIVATE
		cxx_std_17
)

target_compile_definitions(${PROJECT_NAME} PUBLIC
	$<$<CONFIG:Debug>:
		DEBUG_ENABLED
		DEBUG_METHODS_ENABLED
	>
	$<${compiler_is_msvc}:
		TYPED_METHOD_BIND
	>
)

target_link_options(${PROJECT_NAME} PRIVATE
	$<$<NOT:${compiler_is_msvc}>:
		-static-libgcc
		-static-libstdc++
		-Wl,-R,'$$ORIGIN'
	>
)

# Optionally mark headers as SYSTEM
set(GODOT_CPP_SYSTEM_HEADERS_ATTRIBUTE "")
if (GODOT_CPP_SYSTEM_HEADERS)
	set(GODOT_CPP_SYSTEM_HEADERS_ATTRIBUTE SYSTEM)
endif ()

target_include_directories(${PROJECT_NAME} ${GODOT_CPP_SYSTEM_HEADERS_ATTRIBUTE} PUBLIC
	include
	${CMAKE_CURRENT_BINARY_DIR}/gen/include
	${GODOT_GDEXTENSION_DIR}
)

# Add the compile flags
set_property(TARGET ${PROJECT_NAME} APPEND_STRING PROPERTY COMPILE_FLAGS ${GODOT_COMPILE_FLAGS})

# Create the correct name (godot.os.build_type.system_bits)
string(TOLOWER "${CMAKE_SYSTEM_NAME}" SYSTEM_NAME)
string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_TYPE)

if(ANDROID)
	# Added the android abi after system name
	set(SYSTEM_NAME ${SYSTEM_NAME}.${ANDROID_ABI})

	# Android does not have the bits at the end if you look at the main godot repo build
	set(OUTPUT_NAME "godot-cpp.${SYSTEM_NAME}.${BUILD_TYPE}")
else()
	set(OUTPUT_NAME "godot-cpp.${SYSTEM_NAME}.${BUILD_TYPE}.${BITS}")
endif()

set_target_properties(${PROJECT_NAME}
	PROPERTIES
		CXX_EXTENSIONS OFF
		POSITION_INDEPENDENT_CODE ON
		ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
		LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
		OUTPUT_NAME "${OUTPUT_NAME}"
)


[tool.mypy]
disallow_any_generics = true
explicit_package_bases = true
ignore_missing_imports = true
namespace_packages = true
no_implicit_optional = true
pretty = true
scripts_are_modules = true
show_column_numbers = true
warn_redundant_casts = true
warn_return_any = true
warn_unreachable = true

[tool.ruff]
extend-include = ["SConstruct"]
line-length = 120
target-version = "py37"

[tool.ruff.lint]
extend-select = [
	"I", # isort
]

[tool.ruff.lint.per-file-ignores]
"SConstruct" = [
	"F821", # Undefined name
]

[tool.codespell]
enable-colors = ""
write-changes = ""
check-hidden = ""
quiet-level = 3
builtin = "clear,rare,en-GB_to_en-US"
ignore-words-list = """\
	breaked,
	cancelled,
	checkin,
	curvelinear,
	doubleclick,
	expct,
	findn,
	gird,
	hel,
	inout,
	labelin,
	lod,
	mis,
	nd,
	numer,
	ot,
	outin,
	requestor,
	te,
	textin,
	thirdparty,
	vai
"""
#!/usr/bin/env python

import os

EnsureSConsVersion(4, 0)


try:
    Import("env")
except Exception:
    # Default tools with no platform defaults to gnu toolchain.
    # We apply platform specific toolchains via our custom tools.
    env = Environment(tools=["default"], PLATFORM="")

env.PrependENVPath("PATH", os.getenv("PATH"))

# Custom options and profile flags.
customs = ["custom.py"]
try:
    customs += Import("customs")
except Exception:
    pass
profile = ARGUMENTS.get("profile", "")
if profile:
    if os.path.isfile(profile):
        customs.append(profile)
    elif os.path.isfile(profile + ".py"):
        customs.append(profile + ".py")
opts = Variables(customs, ARGUMENTS)
cpp_tool = Tool("godotcpp", toolpath=["tools"])
cpp_tool.options(opts, env)
opts.Update(env)

Help(opts.GenerateHelpText(env))

# Detect and print a warning listing unknown SCons variables to ease troubleshooting.
unknown = opts.UnknownVariables()
if unknown:
    print("WARNING: Unknown SCons variables were passed and will be ignored:")
    for item in unknown.items():
        print("    " + item[0] + "=" + item[1])

scons_cache_path = os.environ.get("SCONS_CACHE")
if scons_cache_path is not None:
    CacheDir(scons_cache_path)
    Decider("MD5")

cpp_tool.generate(env)
library = env.GodotCPP()

Return("env")



/**************************************************************************/
/*  ref.hpp                                                               */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef GODOT_REF_HPP
#define GODOT_REF_HPP

#include <godot_cpp/core/defs.hpp>

#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/binder_common.hpp>
#include <godot_cpp/core/memory.hpp>
#include <godot_cpp/variant/variant.hpp>

namespace godot {

// Helper class for RefCounted objects, same as Godot one.

class RefCounted;

template <typename T>
class Ref {
	T *reference = nullptr;

	void ref(const Ref &p_from) {
		if (p_from.reference == reference) {
			return;
		}

		unref();

		reference = p_from.reference;
		if (reference) {
			reference->reference();
		}
	}

	void ref_pointer(T *p_ref) {
		ERR_FAIL_NULL(p_ref);

		if (p_ref->init_ref()) {
			reference = p_ref;
		}
	}

public:
	_FORCE_INLINE_ bool operator==(const T *p_ptr) const {
		return reference == p_ptr;
	}
	_FORCE_INLINE_ bool operator!=(const T *p_ptr) const {
		return reference != p_ptr;
	}

	_FORCE_INLINE_ bool operator<(const Ref<T> &p_r) const {
		return reference < p_r.reference;
	}
	_FORCE_INLINE_ bool operator==(const Ref<T> &p_r) const {
		return reference == p_r.reference;
	}
	_FORCE_INLINE_ bool operator!=(const Ref<T> &p_r) const {
		return reference != p_r.reference;
	}

	_FORCE_INLINE_ T *operator*() const {
		return reference;
	}

	_FORCE_INLINE_ T *operator->() const {
		return reference;
	}

	_FORCE_INLINE_ T *ptr() const {
		return reference;
	}

	operator Variant() const {
		return Variant(reference);
	}

	void operator=(const Ref &p_from) {
		ref(p_from);
	}

	template <typename T_Other>
	void operator=(const Ref<T_Other> &p_from) {
		RefCounted *refb = const_cast<RefCounted *>(static_cast<const RefCounted *>(p_from.ptr()));
		if (!refb) {
			unref();
			return;
		}

		Ref r;
		r.reference = Object::cast_to<T>(refb);
		ref(r);
		r.reference = nullptr;
	}

	void operator=(const Variant &p_variant) {
		// Needs testing, Variant has a cast to Object * here.

		// Object *object = p_variant.get_validated_object();
		Object *object = p_variant;

		if (object == reference) {
			return;
		}

		unref();

		if (!object) {
			return;
		}

		T *r = Object::cast_to<T>(object);
		if (r && r->reference()) {
			reference = r;
		}
	}

	template <typename T_Other>
	void reference_ptr(T_Other *p_ptr) {
		if (reference == p_ptr) {
			return;
		}
		unref();

		T *r = Object::cast_to<T>(p_ptr);
		if (r) {
			ref_pointer(r);
		}
	}

	Ref(const Ref &p_from) {
		ref(p_from);
	}

	template <typename T_Other>
	Ref(const Ref<T_Other> &p_from) {
		RefCounted *refb = const_cast<RefCounted *>(static_cast<const RefCounted *>(p_from.ptr()));
		if (!refb) {
			unref();
			return;
		}

		Ref r;
		r.reference = Object::cast_to<T>(refb);
		ref(r);
		r.reference = nullptr;
	}

	Ref(T *p_reference) {
		if (p_reference) {
			ref_pointer(p_reference);
		}
	}

	Ref(const Variant &p_variant) {
		// Needs testing, Variant has a cast to Object * here.

		// Object *object = p_variant.get_validated_object();
		Object *object = p_variant;

		if (!object) {
			return;
		}

		T *r = Object::cast_to<T>(object);
		if (r && r->reference()) {
			reference = r;
		}
	}

	inline bool is_valid() const { return reference != nullptr; }
	inline bool is_null() const { return reference == nullptr; }

	void unref() {
		if (reference && reference->unreference()) {
			memdelete(reference);
		}
		reference = nullptr;
	}

	void instantiate() {
		ref(memnew(T()));
	}

	Ref() {}

	~Ref() {
		unref();
	}

	// Used exclusively in the bindings to recreate the Ref Godot encapsulates in return values,
	// without adding to the refcount.
	inline static Ref<T> _gde_internal_constructor(Object *obj) {
		Ref<T> r;
		r.reference = (T *)obj;
		return r;
	}
};

template <typename T>
struct PtrToArg<Ref<T>> {
	_FORCE_INLINE_ static Ref<T> convert(const void *p_ptr) {
		GDExtensionRefPtr ref = (GDExtensionRefPtr)p_ptr;
		ERR_FAIL_NULL_V(p_ptr, Ref<T>());
		return Ref<T>(reinterpret_cast<T *>(godot::internal::get_object_instance_binding(godot::internal::gdextension_interface_ref_get_object(ref))));
	}

	typedef Ref<T> EncodeT;

	_FORCE_INLINE_ static void encode(Ref<T> p_val, void *p_ptr) {
		GDExtensionRefPtr ref = (GDExtensionRefPtr)p_ptr;
		ERR_FAIL_NULL(ref);

		// This code assumes that p_ptr points to an unset Ref<T> variable on the Godot side
		// so we only set it if we have an object to set.
		if (p_val.is_valid()) {
			godot::internal::gdextension_interface_ref_set_object(ref, p_val->_owner);
		}
	}
};

template <typename T>
struct PtrToArg<const Ref<T> &> {
	typedef Ref<T> EncodeT;

	_FORCE_INLINE_ static Ref<T> convert(const void *p_ptr) {
		GDExtensionRefPtr ref = const_cast<GDExtensionRefPtr>(p_ptr);
		ERR_FAIL_NULL_V(p_ptr, Ref<T>());
		return Ref<T>(reinterpret_cast<T *>(godot::internal::get_object_instance_binding(godot::internal::gdextension_interface_ref_get_object(ref))));
	}
};

template <typename T>
struct GetTypeInfo<Ref<T>, typename EnableIf<TypeInherits<RefCounted, T>::value>::type> {
	static constexpr GDExtensionVariantType VARIANT_TYPE = GDEXTENSION_VARIANT_TYPE_OBJECT;
	static constexpr GDExtensionClassMethodArgumentMetadata METADATA = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE;

	static inline PropertyInfo get_class_info() {
		return make_property_info(Variant::Type::OBJECT, "", PROPERTY_HINT_RESOURCE_TYPE, T::get_class_static());
	}
};

template <typename T>
struct GetTypeInfo<const Ref<T> &, typename EnableIf<TypeInherits<RefCounted, T>::value>::type> {
	static constexpr GDExtensionVariantType VARIANT_TYPE = GDEXTENSION_VARIANT_TYPE_OBJECT;
	static constexpr GDExtensionClassMethodArgumentMetadata METADATA = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE;

	static inline PropertyInfo get_class_info() {
		return make_property_info(Variant::Type::OBJECT, "", PROPERTY_HINT_RESOURCE_TYPE, T::get_class_static());
	}
};

} // namespace godot

#endif // GODOT_REF_HPP


#ifndef GODOT_WRAPPED_HPP
#define GODOT_WRAPPED_HPP

#include <godot_cpp/core/memory.hpp>

#include <godot_cpp/core/property_info.hpp>

#include <godot_cpp/templates/list.hpp>
#include <godot_cpp/templates/vector.hpp>

#include <godot_cpp/godot.hpp>

namespace godot {

class ClassDB;

typedef void GodotObject;

template <typename T, std::enable_if_t<std::is_base_of<::godot::Wrapped, T>::value, bool> = true>
_ALWAYS_INLINE_ void _pre_initialize();

// Base for all engine classes, to contain the pointer to the engine instance.
class Wrapped {
	friend class GDExtensionBinding;
	friend class ClassDB;
	friend void postinitialize_handler(Wrapped *);

	template <typename T, std::enable_if_t<std::is_base_of<::godot::Wrapped, T>::value, bool>>
	friend _ALWAYS_INLINE_ void _pre_initialize();

	thread_local static const StringName *_constructing_extension_class_name;
	thread_local static const GDExtensionInstanceBindingCallbacks *_constructing_class_binding_callbacks;

	template <typename T>
	_ALWAYS_INLINE_ static void _set_construct_info() {
		_constructing_extension_class_name = T::_get_extension_class_name();
		_constructing_class_binding_callbacks = &T::_gde_binding_callbacks;
	}

protected:
	virtual bool _is_extension_class() const { return false; }
	static const StringName *_get_extension_class_name(); // This is needed to retrieve the class name before the godot object has its _extension and _extension_instance members assigned.

#ifdef HOT_RELOAD_ENABLED
	struct RecreateInstance {
		GDExtensionClassInstancePtr wrapper;
		GDExtensionObjectPtr owner;
		RecreateInstance *next;
	};
	inline static RecreateInstance *recreate_instance = nullptr;
#endif

	void _notification(int p_what) {}
	bool _set(const StringName &p_name, const Variant &p_property) { return false; }
	bool _get(const StringName &p_name, Variant &r_property) const { return false; }
	void _get_property_list(List<PropertyInfo> *p_list) const {}
	bool _property_can_revert(const StringName &p_name) const { return false; }
	bool _property_get_revert(const StringName &p_name, Variant &r_property) const { return false; }
	void _validate_property(PropertyInfo &p_property) const {}
	String _to_string() const { return "[" + String(get_class_static()) + ":" + itos(get_instance_id()) + "]"; }

	static void notification_bind(GDExtensionClassInstancePtr p_instance, int32_t p_what, GDExtensionBool p_reversed) {}
	static GDExtensionBool set_bind(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionConstVariantPtr p_value) { return false; }
	static GDExtensionBool get_bind(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionVariantPtr r_ret) { return false; }
	static const GDExtensionPropertyInfo *get_property_list_bind(GDExtensionClassInstancePtr p_instance, uint32_t *r_count) { return nullptr; }
	static void free_property_list_bind(GDExtensionClassInstancePtr p_instance, const GDExtensionPropertyInfo *p_list, uint32_t p_count) {}
	static GDExtensionBool property_can_revert_bind(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name) { return false; }
	static GDExtensionBool property_get_revert_bind(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionVariantPtr r_ret) { return false; }
	static GDExtensionBool validate_property_bind(GDExtensionClassInstancePtr p_instance, GDExtensionPropertyInfo *p_property) { return false; }
	static void to_string_bind(GDExtensionClassInstancePtr p_instance, GDExtensionBool *r_is_valid, GDExtensionStringPtr r_out) {}

	// The only reason this has to be held here, is when we return results of `_get_property_list` to Godot, we pass
	// pointers to strings in this list. They have to remain valid to pass the bridge, until the list is freed by Godot...
	::godot::List<::godot::PropertyInfo> plist_owned;

	void _postinitialize();
	virtual void _notificationv(int32_t p_what, bool p_reversed = false) {}

	Wrapped(const StringName p_godot_class);
	Wrapped(GodotObject *p_godot_object);
	virtual ~Wrapped() {}

public:
	static const StringName &get_class_static() {
		static const StringName string_name = StringName("Wrapped");
		return string_name;
	}

	uint64_t get_instance_id() const {
		return 0;
	}

	// Must be public but you should not touch this.
	GodotObject *_owner = nullptr;
};

template <typename T, std::enable_if_t<std::is_base_of<::godot::Wrapped, T>::value, bool>>
_ALWAYS_INLINE_ void _pre_initialize() {
	Wrapped::_set_construct_info<T>();
}

_FORCE_INLINE_ void snarray_add_str(Vector<StringName> &arr) {
}

_FORCE_INLINE_ void snarray_add_str(Vector<StringName> &arr, const StringName &p_str) {
	arr.push_back(p_str);
}

template <typename... P>
_FORCE_INLINE_ void snarray_add_str(Vector<StringName> &arr, const StringName &p_str, P... p_args) {
	arr.push_back(p_str);
	snarray_add_str(arr, p_args...);
}

template <typename... P>
_FORCE_INLINE_ Vector<StringName> snarray(P... p_args) {
	Vector<StringName> arr;
	snarray_add_str(arr, p_args...);
	return arr;
}

namespace internal {

GDExtensionPropertyInfo *create_c_property_list(const ::godot::List<::godot::PropertyInfo> &plist_cpp, uint32_t *r_size);
void free_c_property_list(GDExtensionPropertyInfo *plist);

typedef void (*EngineClassRegistrationCallback)();
void add_engine_class_registration_callback(EngineClassRegistrationCallback p_callback);
void register_engine_class(const StringName &p_name, const GDExtensionInstanceBindingCallbacks *p_callbacks);
void register_engine_classes();

template <typename T>
struct EngineClassRegistration {
	EngineClassRegistration() {
		add_engine_class_registration_callback(&EngineClassRegistration<T>::callback);
	}

	static void callback() {
		register_engine_class(T::get_class_static(), &T::_gde_binding_callbacks);
	}
};

} // namespace internal

} // namespace godot

// Use this on top of your own classes.
// Note: the trail of `***` is to keep sane diffs in PRs, because clang-format otherwise moves every `\` which makes
// every line of the macro different
#define GDCLASS(m_class, m_inherits) /***********************************************************************************************************************************************/ \
private:                                                                                                                                                                               \
	void operator=(const m_class & /*p_rval*/) {}                                                                                                                                      \
	friend class ::godot::ClassDB;                                                                                                                                                     \
	friend class ::godot::Wrapped;                                                                                                                                                     \
                                                                                                                                                                                       \
protected:                                                                                                                                                                             \
	virtual bool _is_extension_class() const override { return true; }                                                                                                                 \
                                                                                                                                                                                       \
	static const ::godot::StringName *_get_extension_class_name() {                                                                                                                    \
		const ::godot::StringName &string_name = get_class_static();                                                                                                                   \
		return &string_name;                                                                                                                                                           \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static void (*_get_bind_methods())() {                                                                                                                                             \
		return &m_class::_bind_methods;                                                                                                                                                \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static void (::godot::Wrapped::*_get_notification())(int) {                                                                                                                        \
		return (void(::godot::Wrapped::*)(int)) & m_class::_notification;                                                                                                              \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static bool (::godot::Wrapped::*_get_set())(const ::godot::StringName &p_name, const ::godot::Variant &p_property) {                                                               \
		return (bool(::godot::Wrapped::*)(const ::godot::StringName &p_name, const ::godot::Variant &p_property)) & m_class::_set;                                                     \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static bool (::godot::Wrapped::*_get_get())(const ::godot::StringName &p_name, ::godot::Variant &r_ret) const {                                                                    \
		return (bool(::godot::Wrapped::*)(const ::godot::StringName &p_name, ::godot::Variant &r_ret) const) & m_class::_get;                                                          \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static void (::godot::Wrapped::*_get_get_property_list())(::godot::List<::godot::PropertyInfo> * p_list) const {                                                                   \
		return (void(::godot::Wrapped::*)(::godot::List<::godot::PropertyInfo> * p_list) const) & m_class::_get_property_list;                                                         \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static bool (::godot::Wrapped::*_get_property_can_revert())(const ::godot::StringName &p_name) const {                                                                             \
		return (bool(::godot::Wrapped::*)(const ::godot::StringName &p_name) const) & m_class::_property_can_revert;                                                                   \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static bool (::godot::Wrapped::*_get_property_get_revert())(const ::godot::StringName &p_name, ::godot::Variant &) const {                                                         \
		return (bool(::godot::Wrapped::*)(const ::godot::StringName &p_name, ::godot::Variant &) const) & m_class::_property_get_revert;                                               \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static void (::godot::Wrapped::*_get_validate_property())(::godot::PropertyInfo & p_property) const {                                                                              \
		return (void(::godot::Wrapped::*)(::godot::PropertyInfo & p_property) const) & m_class::_validate_property;                                                                    \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static ::godot::String (::godot::Wrapped::*_get_to_string())() const {                                                                                                             \
		return (::godot::String(::godot::Wrapped::*)() const) & m_class::_to_string;                                                                                                   \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	template <typename T, typename B>                                                                                                                                                  \
	static void register_virtuals() {                                                                                                                                                  \
		m_inherits::register_virtuals<T, B>();                                                                                                                                         \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
public:                                                                                                                                                                                \
	typedef m_class self_type;                                                                                                                                                         \
	typedef m_inherits parent_type;                                                                                                                                                    \
                                                                                                                                                                                       \
	static void initialize_class() {                                                                                                                                                   \
		static bool initialized = false;                                                                                                                                               \
		if (initialized) {                                                                                                                                                             \
			return;                                                                                                                                                                    \
		}                                                                                                                                                                              \
		m_inherits::initialize_class();                                                                                                                                                \
		if (m_class::_get_bind_methods() != m_inherits::_get_bind_methods()) {                                                                                                         \
			_bind_methods();                                                                                                                                                           \
			m_inherits::register_virtuals<m_class, m_inherits>();                                                                                                                      \
		}                                                                                                                                                                              \
		initialized = true;                                                                                                                                                            \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static const ::godot::StringName &get_class_static() {                                                                                                                             \
		static const ::godot::StringName string_name = ::godot::StringName(#m_class);                                                                                                  \
		return string_name;                                                                                                                                                            \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static const ::godot::StringName &get_parent_class_static() {                                                                                                                      \
		return m_inherits::get_class_static();                                                                                                                                         \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static void notification_bind(GDExtensionClassInstancePtr p_instance, int32_t p_what, GDExtensionBool p_reversed) {                                                                \
		if (p_instance && m_class::_get_notification()) {                                                                                                                              \
			if (!p_reversed) {                                                                                                                                                         \
				m_inherits::notification_bind(p_instance, p_what, p_reversed);                                                                                                         \
			}                                                                                                                                                                          \
			if (m_class::_get_notification() != m_inherits::_get_notification()) {                                                                                                     \
				m_class *cls = reinterpret_cast<m_class *>(p_instance);                                                                                                                \
				cls->_notification(p_what);                                                                                                                                            \
			}                                                                                                                                                                          \
			if (p_reversed) {                                                                                                                                                          \
				m_inherits::notification_bind(p_instance, p_what, p_reversed);                                                                                                         \
			}                                                                                                                                                                          \
		}                                                                                                                                                                              \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static GDExtensionBool set_bind(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionConstVariantPtr p_value) {                                \
		if (p_instance) {                                                                                                                                                              \
			if (m_inherits::set_bind(p_instance, p_name, p_value)) {                                                                                                                   \
				return true;                                                                                                                                                           \
			}                                                                                                                                                                          \
			if (m_class::_get_set() != m_inherits::_get_set()) {                                                                                                                       \
				m_class *cls = reinterpret_cast<m_class *>(p_instance);                                                                                                                \
				return cls->_set(*reinterpret_cast<const ::godot::StringName *>(p_name), *reinterpret_cast<const ::godot::Variant *>(p_value));                                        \
			}                                                                                                                                                                          \
		}                                                                                                                                                                              \
		return false;                                                                                                                                                                  \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static GDExtensionBool get_bind(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionVariantPtr r_ret) {                                       \
		if (p_instance) {                                                                                                                                                              \
			if (m_inherits::get_bind(p_instance, p_name, r_ret)) {                                                                                                                     \
				return true;                                                                                                                                                           \
			}                                                                                                                                                                          \
			if (m_class::_get_get() != m_inherits::_get_get()) {                                                                                                                       \
				m_class *cls = reinterpret_cast<m_class *>(p_instance);                                                                                                                \
				return cls->_get(*reinterpret_cast<const ::godot::StringName *>(p_name), *reinterpret_cast<::godot::Variant *>(r_ret));                                                \
			}                                                                                                                                                                          \
		}                                                                                                                                                                              \
		return false;                                                                                                                                                                  \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static inline bool has_get_property_list() {                                                                                                                                       \
		return m_class::_get_get_property_list() && m_class::_get_get_property_list() != m_inherits::_get_get_property_list();                                                         \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static const GDExtensionPropertyInfo *get_property_list_bind(GDExtensionClassInstancePtr p_instance, uint32_t *r_count) {                                                          \
		if (!p_instance) {                                                                                                                                                             \
			if (r_count)                                                                                                                                                               \
				*r_count = 0;                                                                                                                                                          \
			return nullptr;                                                                                                                                                            \
		}                                                                                                                                                                              \
		m_class *cls = reinterpret_cast<m_class *>(p_instance);                                                                                                                        \
		::godot::List<::godot::PropertyInfo> &plist_cpp = cls->plist_owned;                                                                                                            \
		ERR_FAIL_COND_V_MSG(!plist_cpp.is_empty(), nullptr, "Internal error, property list was not freed by engine!");                                                                 \
		cls->_get_property_list(&plist_cpp);                                                                                                                                           \
		return ::godot::internal::create_c_property_list(plist_cpp, r_count);                                                                                                          \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static void free_property_list_bind(GDExtensionClassInstancePtr p_instance, const GDExtensionPropertyInfo *p_list, uint32_t /*p_count*/) {                                         \
		if (p_instance) {                                                                                                                                                              \
			m_class *cls = reinterpret_cast<m_class *>(p_instance);                                                                                                                    \
			cls->plist_owned.clear();                                                                                                                                                  \
			::godot::internal::free_c_property_list(const_cast<GDExtensionPropertyInfo *>(p_list));                                                                                    \
		}                                                                                                                                                                              \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static GDExtensionBool property_can_revert_bind(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name) {                                                    \
		if (p_instance && m_class::_get_property_can_revert()) {                                                                                                                       \
			if (m_class::_get_property_can_revert() != m_inherits::_get_property_can_revert()) {                                                                                       \
				m_class *cls = reinterpret_cast<m_class *>(p_instance);                                                                                                                \
				return cls->_property_can_revert(*reinterpret_cast<const ::godot::StringName *>(p_name));                                                                              \
			}                                                                                                                                                                          \
			return m_inherits::property_can_revert_bind(p_instance, p_name);                                                                                                           \
		}                                                                                                                                                                              \
		return false;                                                                                                                                                                  \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static GDExtensionBool property_get_revert_bind(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionVariantPtr r_ret) {                       \
		if (p_instance && m_class::_get_property_get_revert()) {                                                                                                                       \
			if (m_class::_get_property_get_revert() != m_inherits::_get_property_get_revert()) {                                                                                       \
				m_class *cls = reinterpret_cast<m_class *>(p_instance);                                                                                                                \
				return cls->_property_get_revert(*reinterpret_cast<const ::godot::StringName *>(p_name), *reinterpret_cast<::godot::Variant *>(r_ret));                                \
			}                                                                                                                                                                          \
			return m_inherits::property_get_revert_bind(p_instance, p_name, r_ret);                                                                                                    \
		}                                                                                                                                                                              \
		return false;                                                                                                                                                                  \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static GDExtensionBool validate_property_bind(GDExtensionClassInstancePtr p_instance, GDExtensionPropertyInfo *p_property) {                                                       \
		bool ret = false;                                                                                                                                                              \
		if (p_instance && m_class::_get_validate_property()) {                                                                                                                         \
			ret = m_inherits::validate_property_bind(p_instance, p_property);                                                                                                          \
			if (m_class::_get_validate_property() != m_inherits::_get_validate_property()) {                                                                                           \
				m_class *cls = reinterpret_cast<m_class *>(p_instance);                                                                                                                \
				::godot::PropertyInfo info(p_property);                                                                                                                                \
				cls->_validate_property(info);                                                                                                                                         \
				info._update(p_property);                                                                                                                                              \
				return true;                                                                                                                                                           \
			}                                                                                                                                                                          \
		}                                                                                                                                                                              \
		return ret;                                                                                                                                                                    \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static void to_string_bind(GDExtensionClassInstancePtr p_instance, GDExtensionBool *r_is_valid, GDExtensionStringPtr r_out) {                                                      \
		if (p_instance && m_class::_get_to_string()) {                                                                                                                                 \
			if (m_class::_get_to_string() != m_inherits::_get_to_string()) {                                                                                                           \
				m_class *cls = reinterpret_cast<m_class *>(p_instance);                                                                                                                \
				*reinterpret_cast<::godot::String *>(r_out) = cls->_to_string();                                                                                                       \
				*r_is_valid = true;                                                                                                                                                    \
				return;                                                                                                                                                                \
			}                                                                                                                                                                          \
			m_inherits::to_string_bind(p_instance, r_is_valid, r_out);                                                                                                                 \
		}                                                                                                                                                                              \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static void free(void * /*data*/, GDExtensionClassInstancePtr ptr) {                                                                                                               \
		if (ptr) {                                                                                                                                                                     \
			m_class *cls = reinterpret_cast<m_class *>(ptr);                                                                                                                           \
			cls->~m_class();                                                                                                                                                           \
			::godot::Memory::free_static(cls);                                                                                                                                         \
		}                                                                                                                                                                              \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static void *_gde_binding_create_callback(void * /*p_token*/, void * /*p_instance*/) {                                                                                             \
		return nullptr;                                                                                                                                                                \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static void _gde_binding_free_callback(void * /*p_token*/, void * /*p_instance*/, void * /*p_binding*/) {                                                                          \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static GDExtensionBool _gde_binding_reference_callback(void * /*p_token*/, void * /*p_instance*/, GDExtensionBool /*p_reference*/) {                                               \
		return true;                                                                                                                                                                   \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static constexpr GDExtensionInstanceBindingCallbacks _gde_binding_callbacks = {                                                                                                    \
		_gde_binding_create_callback,                                                                                                                                                  \
		_gde_binding_free_callback,                                                                                                                                                    \
		_gde_binding_reference_callback,                                                                                                                                               \
	};                                                                                                                                                                                 \
                                                                                                                                                                                       \
protected:                                                                                                                                                                             \
	virtual void _notificationv(int32_t p_what, bool p_reversed = false) override {                                                                                                    \
		m_class::notification_bind(this, p_what, p_reversed);                                                                                                                          \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
private:

// Don't use this for your classes, use GDCLASS() instead.
#define GDEXTENSION_CLASS_ALIAS(m_class, m_alias_for, m_inherits) /******************************************************************************************************************/ \
private:                                                                                                                                                                               \
	inline static ::godot::internal::EngineClassRegistration<m_class> _gde_engine_class_registration_helper;                                                                           \
	void operator=(const m_class &p_rval) {}                                                                                                                                           \
	friend class ::godot::ClassDB;                                                                                                                                                     \
	friend class ::godot::Wrapped;                                                                                                                                                     \
                                                                                                                                                                                       \
protected:                                                                                                                                                                             \
	m_class(const char *p_godot_class) : m_inherits(p_godot_class) {}                                                                                                                  \
	m_class(GodotObject *p_godot_object) : m_inherits(p_godot_object) {}                                                                                                               \
                                                                                                                                                                                       \
	static void _bind_methods() {}                                                                                                                                                     \
                                                                                                                                                                                       \
	static void (*_get_bind_methods())() {                                                                                                                                             \
		return nullptr;                                                                                                                                                                \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static void (Wrapped::*_get_notification())(int) {                                                                                                                                 \
		return nullptr;                                                                                                                                                                \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static bool (Wrapped::*_get_set())(const ::godot::StringName &p_name, const Variant &p_property) {                                                                                 \
		return nullptr;                                                                                                                                                                \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static bool (Wrapped::*_get_get())(const ::godot::StringName &p_name, Variant &r_ret) const {                                                                                      \
		return nullptr;                                                                                                                                                                \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static inline bool has_get_property_list() {                                                                                                                                       \
		return false;                                                                                                                                                                  \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static void (Wrapped::*_get_get_property_list())(List<PropertyInfo> * p_list) const {                                                                                              \
		return nullptr;                                                                                                                                                                \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static bool (Wrapped::*_get_property_can_revert())(const ::godot::StringName &p_name) const {                                                                                      \
		return nullptr;                                                                                                                                                                \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static bool (Wrapped::*_get_property_get_revert())(const ::godot::StringName &p_name, Variant &) const {                                                                           \
		return nullptr;                                                                                                                                                                \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static void (Wrapped::*_get_validate_property())(::godot::PropertyInfo & p_property) const {                                                                                       \
		return nullptr;                                                                                                                                                                \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static String (Wrapped::*_get_to_string())() const {                                                                                                                               \
		return nullptr;                                                                                                                                                                \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
public:                                                                                                                                                                                \
	typedef m_class self_type;                                                                                                                                                         \
	typedef m_inherits parent_type;                                                                                                                                                    \
                                                                                                                                                                                       \
	static void initialize_class() {}                                                                                                                                                  \
                                                                                                                                                                                       \
	static const ::godot::StringName &get_class_static() {                                                                                                                             \
		static const ::godot::StringName string_name = ::godot::StringName(#m_alias_for);                                                                                              \
		return string_name;                                                                                                                                                            \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static const ::godot::StringName &get_parent_class_static() {                                                                                                                      \
		return m_inherits::get_class_static();                                                                                                                                         \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static void free(void *data, GDExtensionClassInstancePtr ptr) {                                                                                                                    \
	}                                                                                                                                                                                  \
                                                                                                                                                                                       \
	static void *_gde_binding_create_callback(void *p_token, void *p_instance) {                                                                                                       \
		/* Do not call memnew here, we don't want the post-initializer to be called */                                                                                                 \
		return new ("", "") m_class((GodotObject *)p_instance);                                                                                                                        \
	}                                                                                                                                                                                  \
	static void _gde_binding_free_callback(void *p_token, void *p_instance, void *p_binding) {                                                                                         \
		/* Explicitly call the deconstructor to ensure proper lifecycle for non-trivial members */                                                                                     \
		reinterpret_cast<m_class *>(p_binding)->~m_class();                                                                                                                            \
		Memory::free_static(reinterpret_cast<m_class *>(p_binding));                                                                                                                   \
	}                                                                                                                                                                                  \
	static GDExtensionBool _gde_binding_reference_callback(void *p_token, void *p_instance, GDExtensionBool p_reference) {                                                             \
		return true;                                                                                                                                                                   \
	}                                                                                                                                                                                  \
	static constexpr GDExtensionInstanceBindingCallbacks _gde_binding_callbacks = {                                                                                                    \
		_gde_binding_create_callback,                                                                                                                                                  \
		_gde_binding_free_callback,                                                                                                                                                    \
		_gde_binding_reference_callback,                                                                                                                                               \
	};                                                                                                                                                                                 \
	m_class() : m_class(#m_alias_for) {}                                                                                                                                               \
                                                                                                                                                                                       \
private:

// Don't use this for your classes, use GDCLASS() instead.
#define GDEXTENSION_CLASS(m_class, m_inherits) GDEXTENSION_CLASS_ALIAS(m_class, m_class, m_inherits)

#define GDVIRTUAL_CALL(m_name, ...) _gdvirtual_##m_name##_call<false>(__VA_ARGS__)
#define GDVIRTUAL_CALL_PTR(m_obj, m_name, ...) m_obj->_gdvirtual_##m_name##_call<false>(__VA_ARGS__)

#define GDVIRTUAL_REQUIRED_CALL(m_name, ...) _gdvirtual_##m_name##_call<true>(__VA_ARGS__)
#define GDVIRTUAL_REQUIRED_CALL_PTR(m_obj, m_name, ...) m_obj->_gdvirtual_##m_name##_call<true>(__VA_ARGS__)

#define GDVIRTUAL_BIND(m_name, ...) ::godot::ClassDB::add_virtual_method(get_class_static(), _gdvirtual_##m_name##_get_method_info(), ::godot::snarray(__VA_ARGS__));
#define GDVIRTUAL_IS_OVERRIDDEN(m_name) _gdvirtual_##m_name##_overridden()
#define GDVIRTUAL_IS_OVERRIDDEN_PTR(m_obj, m_name) m_obj->_gdvirtual_##m_name##_overridden()

#endif // GODOT_WRAPPED_HPP

#ifndef GODOT_ARRAY_HELPERS_HPP
#define GODOT_ARRAY_HELPERS_HPP

namespace godot {
namespace helpers {
template <typename T, typename ValueT>
T append_all(T appendable, ValueT value) {
	appendable.append(value);
	return appendable;
}

template <typename T, typename ValueT, typename... Args>
T append_all(T appendable, ValueT value, Args... args) {
	appendable.append(value);
	return append_all(appendable, args...);
}

template <typename T>
T append_all(T appendable) {
	return appendable;
}
} // namespace helpers
} // namespace godot

#endif // GODOT_ARRAY_HELPERS_HPP




#ifndef GODOT_VECTOR2I_HPP
#define GODOT_VECTOR2I_HPP

#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/core/math.hpp>

namespace godot {

class String;
struct Vector2;

struct _NO_DISCARD_ Vector2i {
	static const int AXIS_COUNT = 2;

	enum Axis {
		AXIS_X,
		AXIS_Y,
	};

	union {
		struct {
			union {
				int32_t x;
				int32_t width;
			};
			union {
				int32_t y;
				int32_t height;
			};
		};

		int32_t coord[2] = { 0 };
	};

	_FORCE_INLINE_ int32_t &operator[](int p_idx) {
		DEV_ASSERT((unsigned int)p_idx < 2);
		return coord[p_idx];
	}
	_FORCE_INLINE_ const int32_t &operator[](int p_idx) const {
		DEV_ASSERT((unsigned int)p_idx < 2);
		return coord[p_idx];
	}

	_FORCE_INLINE_ Vector2i::Axis min_axis_index() const {
		return x < y ? Vector2i::AXIS_X : Vector2i::AXIS_Y;
	}

	_FORCE_INLINE_ Vector2i::Axis max_axis_index() const {
		return x < y ? Vector2i::AXIS_Y : Vector2i::AXIS_X;
	}

	Vector2i min(const Vector2i &p_vector2i) const {
		return Vector2i(MIN(x, p_vector2i.x), MIN(y, p_vector2i.y));
	}

	Vector2i mini(int32_t p_scalar) const {
		return Vector2i(MIN(x, p_scalar), MIN(y, p_scalar));
	}

	Vector2i max(const Vector2i &p_vector2i) const {
		return Vector2i(MAX(x, p_vector2i.x), MAX(y, p_vector2i.y));
	}

	Vector2i maxi(int32_t p_scalar) const {
		return Vector2i(MAX(x, p_scalar), MAX(y, p_scalar));
	}

	Vector2i operator+(const Vector2i &p_v) const;
	void operator+=(const Vector2i &p_v);
	Vector2i operator-(const Vector2i &p_v) const;
	void operator-=(const Vector2i &p_v);
	Vector2i operator*(const Vector2i &p_v1) const;

	Vector2i operator*(const int32_t &rvalue) const;
	void operator*=(const int32_t &rvalue);

	Vector2i operator/(const Vector2i &p_v1) const;
	Vector2i operator/(const int32_t &rvalue) const;
	void operator/=(const int32_t &rvalue);

	Vector2i operator%(const Vector2i &p_v1) const;
	Vector2i operator%(const int32_t &rvalue) const;
	void operator%=(const int32_t &rvalue);

	Vector2i operator-() const;
	bool operator<(const Vector2i &p_vec2) const { return (x == p_vec2.x) ? (y < p_vec2.y) : (x < p_vec2.x); }
	bool operator>(const Vector2i &p_vec2) const { return (x == p_vec2.x) ? (y > p_vec2.y) : (x > p_vec2.x); }

	bool operator<=(const Vector2i &p_vec2) const { return x == p_vec2.x ? (y <= p_vec2.y) : (x < p_vec2.x); }
	bool operator>=(const Vector2i &p_vec2) const { return x == p_vec2.x ? (y >= p_vec2.y) : (x > p_vec2.x); }

	bool operator==(const Vector2i &p_vec2) const;
	bool operator!=(const Vector2i &p_vec2) const;

	int64_t length_squared() const;
	double length() const;

	int64_t distance_squared_to(const Vector2i &p_to) const;
	double distance_to(const Vector2i &p_to) const;

	real_t aspect() const { return width / (real_t)height; }
	Vector2i sign() const { return Vector2i(SIGN(x), SIGN(y)); }
	Vector2i abs() const { return Vector2i(Math::abs(x), Math::abs(y)); }
	Vector2i snapped(const Vector2i &p_step) const;
	Vector2i snappedi(int32_t p_step) const;
	Vector2i clamp(const Vector2i &p_min, const Vector2i &p_max) const;
	Vector2i clampi(int32_t p_min, int32_t p_max) const;

	operator String() const;
	operator Vector2() const;

	inline Vector2i() {}
	inline Vector2i(const int32_t p_x, const int32_t p_y) {
		x = p_x;
		y = p_y;
	}
};

// Multiplication operators required to workaround issues with LLVM using implicit conversion.

_FORCE_INLINE_ Vector2i operator*(const int32_t p_scalar, const Vector2i &p_vector) {
	return p_vector * p_scalar;
}

_FORCE_INLINE_ Vector2i operator*(const int64_t p_scalar, const Vector2i &p_vector) {
	return p_vector * p_scalar;
}

_FORCE_INLINE_ Vector2i operator*(const float p_scalar, const Vector2i &p_vector) {
	return p_vector * p_scalar;
}

_FORCE_INLINE_ Vector2i operator*(const double p_scalar, const Vector2i &p_vector) {
	return p_vector * p_scalar;
}

typedef Vector2i Size2i;
typedef Vector2i Point2i;

} // namespace godot

#endif // GODOT_VECTOR2I_HPP

#ifndef GODOT_TYPE_INFO_HPP
#define GODOT_TYPE_INFO_HPP

#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <gdextension_interface.h>

namespace godot {

template <bool C, typename T = void>
struct EnableIf {
	typedef T type;
};

template <typename T>
struct EnableIf<false, T> {
};

template <typename, typename>
struct TypesAreSame {
	static bool const value = false;
};

template <typename A>
struct TypesAreSame<A, A> {
	static bool const value = true;
};

template <auto A, auto B>
struct FunctionsAreSame {
	static bool const value = false;
};

template <auto A>
struct FunctionsAreSame<A, A> {
	static bool const value = true;
};

template <typename B, typename D>
struct TypeInherits {
	static D *get_d();

	static char (&test(B *))[1];
	static char (&test(...))[2];

	static bool const value = sizeof(test(get_d())) == sizeof(char) &&
			!TypesAreSame<B volatile const, void volatile const>::value;
};

static PropertyInfo make_property_info(Variant::Type p_type, const StringName &p_name, uint32_t p_hint = PROPERTY_HINT_NONE, const String &p_hint_string = "", uint32_t p_usage = PROPERTY_USAGE_DEFAULT, const StringName &p_class_name = "") {
	PropertyInfo info;
	info.type = p_type;
	info.name = p_name;
	info.hint = p_hint;
	info.hint_string = p_hint_string;
	info.usage = p_usage;
	if (p_hint == PROPERTY_HINT_RESOURCE_TYPE) {
		info.class_name = p_hint_string;
	} else {
		info.class_name = p_class_name;
	}
	return info;
}

// If the compiler fails because it's trying to instantiate the primary 'GetTypeInfo' template
// instead of one of the specializations, it's most likely because the type 'T' is not supported.
// If 'T' is a class that inherits 'Object', make sure it can see the actual class declaration
// instead of a forward declaration. You can always forward declare 'T' in a header file, and then
// include the actual declaration of 'T' in the source file where 'GetTypeInfo<T>' is instantiated.

template <typename T, typename = void>
struct GetTypeInfo;

#define MAKE_TYPE_INFO(m_type, m_var_type)                                                                            \
	template <>                                                                                                       \
	struct GetTypeInfo<m_type> {                                                                                      \
		static constexpr GDExtensionVariantType VARIANT_TYPE = m_var_type;                                            \
		static constexpr GDExtensionClassMethodArgumentMetadata METADATA = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE; \
		static inline PropertyInfo get_class_info() {                                                                 \
			return make_property_info((Variant::Type)VARIANT_TYPE, "");                                               \
		}                                                                                                             \
	};                                                                                                                \
	template <>                                                                                                       \
	struct GetTypeInfo<const m_type &> {                                                                              \
		static constexpr GDExtensionVariantType VARIANT_TYPE = m_var_type;                                            \
		static constexpr GDExtensionClassMethodArgumentMetadata METADATA = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE; \
		static inline PropertyInfo get_class_info() {                                                                 \
			return make_property_info((Variant::Type)VARIANT_TYPE, "");                                               \
		}                                                                                                             \
	};

#define MAKE_TYPE_INFO_WITH_META(m_type, m_var_type, m_metadata)                       \
	template <>                                                                        \
	struct GetTypeInfo<m_type> {                                                       \
		static constexpr GDExtensionVariantType VARIANT_TYPE = m_var_type;             \
		static constexpr GDExtensionClassMethodArgumentMetadata METADATA = m_metadata; \
		static inline PropertyInfo get_class_info() {                                  \
			return make_property_info((Variant::Type)VARIANT_TYPE, "");                \
		}                                                                              \
	};                                                                                 \
	template <>                                                                        \
	struct GetTypeInfo<const m_type &> {                                               \
		static constexpr GDExtensionVariantType VARIANT_TYPE = m_var_type;             \
		static constexpr GDExtensionClassMethodArgumentMetadata METADATA = m_metadata; \
		static inline PropertyInfo get_class_info() {                                  \
			return make_property_info((Variant::Type)VARIANT_TYPE, "");                \
		}                                                                              \
	};

MAKE_TYPE_INFO(bool, GDEXTENSION_VARIANT_TYPE_BOOL)
MAKE_TYPE_INFO_WITH_META(uint8_t, GDEXTENSION_VARIANT_TYPE_INT, GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_UINT8)
MAKE_TYPE_INFO_WITH_META(int8_t, GDEXTENSION_VARIANT_TYPE_INT, GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_INT8)
MAKE_TYPE_INFO_WITH_META(uint16_t, GDEXTENSION_VARIANT_TYPE_INT, GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_UINT16)
MAKE_TYPE_INFO_WITH_META(int16_t, GDEXTENSION_VARIANT_TYPE_INT, GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_INT16)
MAKE_TYPE_INFO_WITH_META(uint32_t, GDEXTENSION_VARIANT_TYPE_INT, GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_UINT32)
MAKE_TYPE_INFO_WITH_META(int32_t, GDEXTENSION_VARIANT_TYPE_INT, GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_INT32)
MAKE_TYPE_INFO_WITH_META(uint64_t, GDEXTENSION_VARIANT_TYPE_INT, GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_UINT64)
MAKE_TYPE_INFO_WITH_META(int64_t, GDEXTENSION_VARIANT_TYPE_INT, GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_INT64)
MAKE_TYPE_INFO(char16_t, GDEXTENSION_VARIANT_TYPE_INT)
MAKE_TYPE_INFO(char32_t, GDEXTENSION_VARIANT_TYPE_INT)
MAKE_TYPE_INFO_WITH_META(float, GDEXTENSION_VARIANT_TYPE_FLOAT, GDEXTENSION_METHOD_ARGUMENT_METADATA_REAL_IS_FLOAT)
MAKE_TYPE_INFO_WITH_META(double, GDEXTENSION_VARIANT_TYPE_FLOAT, GDEXTENSION_METHOD_ARGUMENT_METADATA_REAL_IS_DOUBLE)

MAKE_TYPE_INFO(String, GDEXTENSION_VARIANT_TYPE_STRING)
MAKE_TYPE_INFO(Vector2, GDEXTENSION_VARIANT_TYPE_VECTOR2)
MAKE_TYPE_INFO(Vector2i, GDEXTENSION_VARIANT_TYPE_VECTOR2I)
MAKE_TYPE_INFO(Rect2, GDEXTENSION_VARIANT_TYPE_RECT2)
MAKE_TYPE_INFO(Rect2i, GDEXTENSION_VARIANT_TYPE_RECT2I)
MAKE_TYPE_INFO(Vector3, GDEXTENSION_VARIANT_TYPE_VECTOR3)
MAKE_TYPE_INFO(Vector3i, GDEXTENSION_VARIANT_TYPE_VECTOR3I)
MAKE_TYPE_INFO(Transform2D, GDEXTENSION_VARIANT_TYPE_TRANSFORM2D)
MAKE_TYPE_INFO(Vector4, GDEXTENSION_VARIANT_TYPE_VECTOR4)
MAKE_TYPE_INFO(Vector4i, GDEXTENSION_VARIANT_TYPE_VECTOR4I)
MAKE_TYPE_INFO(Plane, GDEXTENSION_VARIANT_TYPE_PLANE)
MAKE_TYPE_INFO(Quaternion, GDEXTENSION_VARIANT_TYPE_QUATERNION)
MAKE_TYPE_INFO(AABB, GDEXTENSION_VARIANT_TYPE_AABB)
MAKE_TYPE_INFO(Basis, GDEXTENSION_VARIANT_TYPE_BASIS)
MAKE_TYPE_INFO(Transform3D, GDEXTENSION_VARIANT_TYPE_TRANSFORM3D)
MAKE_TYPE_INFO(Projection, GDEXTENSION_VARIANT_TYPE_PROJECTION)
MAKE_TYPE_INFO(Color, GDEXTENSION_VARIANT_TYPE_COLOR)
MAKE_TYPE_INFO(StringName, GDEXTENSION_VARIANT_TYPE_STRING_NAME)
MAKE_TYPE_INFO(NodePath, GDEXTENSION_VARIANT_TYPE_NODE_PATH)
MAKE_TYPE_INFO(RID, GDEXTENSION_VARIANT_TYPE_RID)
MAKE_TYPE_INFO(Callable, GDEXTENSION_VARIANT_TYPE_CALLABLE)
MAKE_TYPE_INFO(Signal, GDEXTENSION_VARIANT_TYPE_SIGNAL)
MAKE_TYPE_INFO(Dictionary, GDEXTENSION_VARIANT_TYPE_DICTIONARY)
MAKE_TYPE_INFO(Array, GDEXTENSION_VARIANT_TYPE_ARRAY)
MAKE_TYPE_INFO(PackedByteArray, GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY)
MAKE_TYPE_INFO(PackedInt32Array, GDEXTENSION_VARIANT_TYPE_PACKED_INT32_ARRAY)
MAKE_TYPE_INFO(PackedInt64Array, GDEXTENSION_VARIANT_TYPE_PACKED_INT64_ARRAY)
MAKE_TYPE_INFO(PackedFloat32Array, GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY)
MAKE_TYPE_INFO(PackedFloat64Array, GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT64_ARRAY)
MAKE_TYPE_INFO(PackedStringArray, GDEXTENSION_VARIANT_TYPE_PACKED_STRING_ARRAY)
MAKE_TYPE_INFO(PackedVector2Array, GDEXTENSION_VARIANT_TYPE_PACKED_VECTOR2_ARRAY)
MAKE_TYPE_INFO(PackedVector3Array, GDEXTENSION_VARIANT_TYPE_PACKED_VECTOR3_ARRAY)
MAKE_TYPE_INFO(PackedVector4Array, GDEXTENSION_VARIANT_TYPE_PACKED_VECTOR4_ARRAY)
MAKE_TYPE_INFO(PackedColorArray, GDEXTENSION_VARIANT_TYPE_PACKED_COLOR_ARRAY)

// For variant.
template <>
struct GetTypeInfo<Variant> {
	static constexpr GDExtensionVariantType VARIANT_TYPE = GDEXTENSION_VARIANT_TYPE_NIL;
	static constexpr GDExtensionClassMethodArgumentMetadata METADATA = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE;
	static inline PropertyInfo get_class_info() {
		return make_property_info(Variant::Type::NIL, "", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NIL_IS_VARIANT);
	}
};

template <>
struct GetTypeInfo<const Variant &> {
	static constexpr GDExtensionVariantType VARIANT_TYPE = GDEXTENSION_VARIANT_TYPE_NIL;
	static constexpr GDExtensionClassMethodArgumentMetadata METADATA = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE;
	static inline PropertyInfo get_class_info() {
		return make_property_info(Variant::Type::NIL, "", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NIL_IS_VARIANT);
	}
};

template <typename T>
struct GetTypeInfo<T *, typename EnableIf<TypeInherits<Object, T>::value>::type> {
	static constexpr GDExtensionVariantType VARIANT_TYPE = GDEXTENSION_VARIANT_TYPE_OBJECT;
	static constexpr GDExtensionClassMethodArgumentMetadata METADATA = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE;
	static inline PropertyInfo get_class_info() {
		return make_property_info(Variant::Type::OBJECT, "", PROPERTY_HINT_RESOURCE_TYPE, T::get_class_static());
	}
};

template <typename T>
struct GetTypeInfo<const T *, typename EnableIf<TypeInherits<Object, T>::value>::type> {
	static constexpr GDExtensionVariantType VARIANT_TYPE = GDEXTENSION_VARIANT_TYPE_OBJECT;
	static constexpr GDExtensionClassMethodArgumentMetadata METADATA = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE;
	static inline PropertyInfo get_class_info() {
		return make_property_info(Variant::Type::OBJECT, "", PROPERTY_HINT_RESOURCE_TYPE, T::get_class_static());
	}
};

inline String enum_qualified_name_to_class_info_name(const String &p_qualified_name) {
	PackedStringArray parts = p_qualified_name.split("::", false);
	if (parts.size() <= 2) {
		return String(".").join(parts);
	}
	// Contains namespace. We only want the class and enum names.
	return parts[parts.size() - 2] + "." + parts[parts.size() - 1];
}

#define TEMPL_MAKE_ENUM_TYPE_INFO(m_enum, m_impl)                                                                                            \
	template <>                                                                                                                              \
	struct GetTypeInfo<m_impl> {                                                                                                             \
		static constexpr Variant::Type VARIANT_TYPE = Variant::INT;                                                                          \
		static constexpr GDExtensionClassMethodArgumentMetadata METADATA = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE;                        \
		static inline PropertyInfo get_class_info() {                                                                                        \
			return make_property_info(Variant::Type::INT, "", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_CLASS_IS_ENUM, \
					enum_qualified_name_to_class_info_name(#m_enum));                                                                        \
		}                                                                                                                                    \
	};

#define MAKE_ENUM_TYPE_INFO(m_enum)                 \
	TEMPL_MAKE_ENUM_TYPE_INFO(m_enum, m_enum)       \
	TEMPL_MAKE_ENUM_TYPE_INFO(m_enum, m_enum const) \
	TEMPL_MAKE_ENUM_TYPE_INFO(m_enum, m_enum &)     \
	TEMPL_MAKE_ENUM_TYPE_INFO(m_enum, const m_enum &)

template <typename T>
inline StringName _gde_constant_get_enum_name(T param, StringName p_constant) {
	if (GetTypeInfo<T>::VARIANT_TYPE == Variant::NIL) {
		ERR_PRINT(("Missing VARIANT_ENUM_CAST for constant's enum: " + String(p_constant)).utf8().get_data());
	}
	return GetTypeInfo<T>::get_class_info().class_name;
}

template <typename T>
class BitField {
	int64_t value = 0;

public:
	_FORCE_INLINE_ void set_flag(T p_flag) { value |= p_flag; }
	_FORCE_INLINE_ bool has_flag(T p_flag) const { return value & p_flag; }
	_FORCE_INLINE_ void clear_flag(T p_flag) { value &= ~p_flag; }
	_FORCE_INLINE_ BitField(int64_t p_value) { value = p_value; }
	_FORCE_INLINE_ operator int64_t() const { return value; }
	_FORCE_INLINE_ operator Variant() const { return value; }
};

#define TEMPL_MAKE_BITFIELD_TYPE_INFO(m_enum, m_impl)                                                                                            \
	template <>                                                                                                                                  \
	struct GetTypeInfo<m_impl> {                                                                                                                 \
		static constexpr Variant::Type VARIANT_TYPE = Variant::INT;                                                                              \
		static constexpr GDExtensionClassMethodArgumentMetadata METADATA = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE;                            \
		static inline PropertyInfo get_class_info() {                                                                                            \
			return make_property_info(Variant::Type::INT, "", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_CLASS_IS_BITFIELD, \
					enum_qualified_name_to_class_info_name(#m_enum));                                                                            \
		}                                                                                                                                        \
	};                                                                                                                                           \
	template <>                                                                                                                                  \
	struct GetTypeInfo<BitField<m_impl>> {                                                                                                       \
		static constexpr Variant::Type VARIANT_TYPE = Variant::INT;                                                                              \
		static constexpr GDExtensionClassMethodArgumentMetadata METADATA = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE;                            \
		static inline PropertyInfo get_class_info() {                                                                                            \
			return make_property_info(Variant::Type::INT, "", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_CLASS_IS_BITFIELD, \
					enum_qualified_name_to_class_info_name(#m_enum));                                                                            \
		}                                                                                                                                        \
	};

#define MAKE_BITFIELD_TYPE_INFO(m_enum)                 \
	TEMPL_MAKE_BITFIELD_TYPE_INFO(m_enum, m_enum)       \
	TEMPL_MAKE_BITFIELD_TYPE_INFO(m_enum, m_enum const) \
	TEMPL_MAKE_BITFIELD_TYPE_INFO(m_enum, m_enum &)     \
	TEMPL_MAKE_BITFIELD_TYPE_INFO(m_enum, const m_enum &)

template <typename T>
inline StringName _gde_constant_get_bitfield_name(T param, StringName p_constant) {
	if (GetTypeInfo<T>::VARIANT_TYPE == Variant::NIL) {
		ERR_PRINT(("Missing VARIANT_ENUM_CAST for constant's bitfield: " + String(p_constant)).utf8().get_data());
	}
	return GetTypeInfo<BitField<T>>::get_class_info().class_name;
}

template <typename T>
struct PtrToArg<TypedArray<T>> {
	_FORCE_INLINE_ static TypedArray<T> convert(const void *p_ptr) {
		return TypedArray<T>(*reinterpret_cast<const Array *>(p_ptr));
	}
	typedef Array EncodeT;
	_FORCE_INLINE_ static void encode(TypedArray<T> p_val, void *p_ptr) {
		*reinterpret_cast<Array *>(p_ptr) = p_val;
	}
};

template <typename T>
struct PtrToArg<const TypedArray<T> &> {
	typedef Array EncodeT;
	_FORCE_INLINE_ static TypedArray<T>
	convert(const void *p_ptr) {
		return TypedArray<T>(*reinterpret_cast<const Array *>(p_ptr));
	}
};

template <typename T>
struct GetTypeInfo<TypedArray<T>> {
	static constexpr GDExtensionVariantType VARIANT_TYPE = GDEXTENSION_VARIANT_TYPE_ARRAY;
	static constexpr GDExtensionClassMethodArgumentMetadata METADATA = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE;
	static inline PropertyInfo get_class_info() {
		return make_property_info(Variant::Type::ARRAY, "", PROPERTY_HINT_ARRAY_TYPE, T::get_class_static());
	}
};

template <typename T>
struct GetTypeInfo<const TypedArray<T> &> {
	static constexpr GDExtensionVariantType VARIANT_TYPE = GDEXTENSION_VARIANT_TYPE_ARRAY;
	static constexpr GDExtensionClassMethodArgumentMetadata METADATA = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE;
	static inline PropertyInfo get_class_info() {
		return make_property_info(Variant::Type::ARRAY, "", PROPERTY_HINT_ARRAY_TYPE, T::get_class_static());
	}
};

#define MAKE_TYPED_ARRAY_INFO(m_type, m_variant_type)                                                                                                \
	template <>                                                                                                                                      \
	struct GetTypeInfo<TypedArray<m_type>> {                                                                                                         \
		static constexpr GDExtensionVariantType VARIANT_TYPE = GDEXTENSION_VARIANT_TYPE_ARRAY;                                                       \
		static constexpr GDExtensionClassMethodArgumentMetadata METADATA = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE;                                \
		static inline PropertyInfo get_class_info() {                                                                                                \
			return make_property_info(Variant::Type::ARRAY, "", PROPERTY_HINT_ARRAY_TYPE, Variant::get_type_name(m_variant_type).utf8().get_data()); \
		}                                                                                                                                            \
	};                                                                                                                                               \
	template <>                                                                                                                                      \
	struct GetTypeInfo<const TypedArray<m_type> &> {                                                                                                 \
		static constexpr GDExtensionVariantType VARIANT_TYPE = GDEXTENSION_VARIANT_TYPE_ARRAY;                                                       \
		static constexpr GDExtensionClassMethodArgumentMetadata METADATA = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE;                                \
		static inline PropertyInfo get_class_info() {                                                                                                \
			return make_property_info(Variant::Type::ARRAY, "", PROPERTY_HINT_ARRAY_TYPE, Variant::get_type_name(m_variant_type).utf8().get_data()); \
		}                                                                                                                                            \
	};

MAKE_TYPED_ARRAY_INFO(bool, Variant::BOOL)
MAKE_TYPED_ARRAY_INFO(uint8_t, Variant::INT)
MAKE_TYPED_ARRAY_INFO(int8_t, Variant::INT)
MAKE_TYPED_ARRAY_INFO(uint16_t, Variant::INT)
MAKE_TYPED_ARRAY_INFO(int16_t, Variant::INT)
MAKE_TYPED_ARRAY_INFO(uint32_t, Variant::INT)
MAKE_TYPED_ARRAY_INFO(int32_t, Variant::INT)
MAKE_TYPED_ARRAY_INFO(uint64_t, Variant::INT)
MAKE_TYPED_ARRAY_INFO(int64_t, Variant::INT)
MAKE_TYPED_ARRAY_INFO(float, Variant::FLOAT)
MAKE_TYPED_ARRAY_INFO(double, Variant::FLOAT)
MAKE_TYPED_ARRAY_INFO(String, Variant::STRING)
MAKE_TYPED_ARRAY_INFO(Vector2, Variant::VECTOR2)
MAKE_TYPED_ARRAY_INFO(Vector2i, Variant::VECTOR2I)
MAKE_TYPED_ARRAY_INFO(Rect2, Variant::RECT2)
MAKE_TYPED_ARRAY_INFO(Rect2i, Variant::RECT2I)
MAKE_TYPED_ARRAY_INFO(Vector3, Variant::VECTOR3)
MAKE_TYPED_ARRAY_INFO(Vector3i, Variant::VECTOR3I)
MAKE_TYPED_ARRAY_INFO(Transform2D, Variant::TRANSFORM2D)
MAKE_TYPED_ARRAY_INFO(Vector4, Variant::VECTOR4)
MAKE_TYPED_ARRAY_INFO(Vector4i, Variant::VECTOR4I)
MAKE_TYPED_ARRAY_INFO(Plane, Variant::PLANE)
MAKE_TYPED_ARRAY_INFO(Quaternion, Variant::QUATERNION)
MAKE_TYPED_ARRAY_INFO(AABB, Variant::AABB)
MAKE_TYPED_ARRAY_INFO(Basis, Variant::BASIS)
MAKE_TYPED_ARRAY_INFO(Transform3D, Variant::TRANSFORM3D)
MAKE_TYPED_ARRAY_INFO(Projection, Variant::PROJECTION)
MAKE_TYPED_ARRAY_INFO(Color, Variant::COLOR)
MAKE_TYPED_ARRAY_INFO(StringName, Variant::STRING_NAME)
MAKE_TYPED_ARRAY_INFO(NodePath, Variant::NODE_PATH)
MAKE_TYPED_ARRAY_INFO(RID, Variant::RID)
MAKE_TYPED_ARRAY_INFO(Callable, Variant::CALLABLE)
MAKE_TYPED_ARRAY_INFO(Signal, Variant::SIGNAL)
MAKE_TYPED_ARRAY_INFO(Dictionary, Variant::DICTIONARY)
MAKE_TYPED_ARRAY_INFO(Array, Variant::ARRAY)
/*
MAKE_TYPED_ARRAY_INFO(Vector<uint8_t>, Variant::PACKED_BYTE_ARRAY)
MAKE_TYPED_ARRAY_INFO(Vector<int32_t>, Variant::PACKED_INT32_ARRAY)
MAKE_TYPED_ARRAY_INFO(Vector<int64_t>, Variant::PACKED_INT64_ARRAY)
MAKE_TYPED_ARRAY_INFO(Vector<float>, Variant::PACKED_FLOAT32_ARRAY)
MAKE_TYPED_ARRAY_INFO(Vector<double>, Variant::PACKED_FLOAT64_ARRAY)
MAKE_TYPED_ARRAY_INFO(Vector<String>, Variant::PACKED_STRING_ARRAY)
MAKE_TYPED_ARRAY_INFO(Vector<Vector2>, Variant::PACKED_VECTOR2_ARRAY)
MAKE_TYPED_ARRAY_INFO(Vector<Vector3>, Variant::PACKED_VECTOR3_ARRAY)
MAKE_TYPED_ARRAY_INFO(Vector<Color>, Variant::PACKED_COLOR_ARRAY)
MAKE_TYPED_ARRAY_INFO(IPAddress, Variant::STRING)
*/

#undef MAKE_TYPED_ARRAY_INFO

#define CLASS_INFO(m_type) (GetTypeInfo<m_type *>::get_class_info())

} // namespace godot

#endif // GODOT_TYPE_INFO_HPP

#ifndef GODOT_CLASS_DB_HPP
#define GODOT_CLASS_DB_HPP

#include <gdextension_interface.h>

#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/core/method_bind.hpp>
#include <godot_cpp/core/object.hpp>

#include <godot_cpp/classes/class_db_singleton.hpp>

// Makes callable_mp readily available in all classes connecting signals.
// Needs to come after method_bind and object have been included.
#include <godot_cpp/variant/callable_method_pointer.hpp>

#include <list>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// Needed to use StringName as key in `std::unordered_map`
template <>
struct std::hash<godot::StringName> {
	std::size_t operator()(godot::StringName const &s) const noexcept {
		return s.hash();
	}
};

namespace godot {

#define DEFVAL(m_defval) (m_defval)

struct MethodDefinition {
	StringName name;
	std::list<StringName> args;
	MethodDefinition() {}
	MethodDefinition(StringName p_name) :
			name(p_name) {}
};

MethodDefinition D_METHOD(StringName p_name);
MethodDefinition D_METHOD(StringName p_name, StringName p_arg1);
template <typename... Args>
MethodDefinition D_METHOD(StringName p_name, StringName p_arg1, Args... args) {
	MethodDefinition md = D_METHOD(p_name, args...);
	md.args.push_front(p_arg1);
	return md;
}

class ClassDB {
	static GDExtensionInitializationLevel current_level;

	friend class godot::GDExtensionBinding;

public:
	struct ClassInfo {
		StringName name;
		StringName parent_name;
		GDExtensionInitializationLevel level = GDEXTENSION_INITIALIZATION_SCENE;
		std::unordered_map<StringName, MethodBind *> method_map;
		std::set<StringName> signal_names;
		std::unordered_map<StringName, GDExtensionClassCallVirtual> virtual_methods;
		std::set<StringName> property_names;
		std::set<StringName> constant_names;
		// Pointer to the parent custom class, if any. Will be null if the parent class is a Godot class.
		ClassInfo *parent_ptr = nullptr;
	};

private:
	// This may only contain custom classes, not Godot classes
	static std::unordered_map<StringName, ClassInfo> classes;
	static std::unordered_map<StringName, const GDExtensionInstanceBindingCallbacks *> instance_binding_callbacks;
	// Used to remember the custom class registration order.
	static std::vector<StringName> class_register_order;
	static std::unordered_map<StringName, Object *> engine_singletons;
	static std::mutex engine_singletons_mutex;

	static MethodBind *bind_methodfi(uint32_t p_flags, MethodBind *p_bind, const MethodDefinition &method_name, const void **p_defs, int p_defcount);
	static void initialize_class(const ClassInfo &cl);
	static void bind_method_godot(const StringName &p_class_name, MethodBind *p_method);

	template <typename T, bool is_abstract>
	static void _register_class(bool p_virtual = false, bool p_exposed = true, bool p_runtime = false);

	template <typename T>
	static GDExtensionObjectPtr _create_instance_func(void *data) {
		if constexpr (!std::is_abstract_v<T>) {
			T *new_object = memnew(T);
			return new_object->_owner;
		} else {
			return nullptr;
		}
	}

	template <typename T>
	static GDExtensionClassInstancePtr _recreate_instance_func(void *data, GDExtensionObjectPtr obj) {
		if constexpr (!std::is_abstract_v<T>) {
#ifdef HOT_RELOAD_ENABLED
			T *new_instance = (T *)memalloc(sizeof(T));
			Wrapped::RecreateInstance recreate_data = { new_instance, obj, Wrapped::recreate_instance };
			Wrapped::recreate_instance = &recreate_data;
			memnew_placement(new_instance, T);
			return new_instance;
#else
			return nullptr;
#endif
		} else {
			return nullptr;
		}
	}

public:
	template <typename T>
	static void register_class(bool p_virtual = false);
	template <typename T>
	static void register_abstract_class();
	template <typename T>
	static void register_internal_class();
	template <typename T>
	static void register_runtime_class();

	_FORCE_INLINE_ static void _register_engine_class(const StringName &p_name, const GDExtensionInstanceBindingCallbacks *p_callbacks) {
		instance_binding_callbacks[p_name] = p_callbacks;
	}

	static void _register_engine_singleton(const StringName &p_class_name, Object *p_singleton) {
		std::lock_guard<std::mutex> lock(engine_singletons_mutex);
		std::unordered_map<StringName, Object *>::const_iterator i = engine_singletons.find(p_class_name);
		if (i != engine_singletons.end()) {
			ERR_FAIL_COND((*i).second != p_singleton);
			return;
		}
		engine_singletons[p_class_name] = p_singleton;
	}

	static void _unregister_engine_singleton(const StringName &p_class_name) {
		std::lock_guard<std::mutex> lock(engine_singletons_mutex);
		engine_singletons.erase(p_class_name);
	}

	template <typename N, typename M, typename... VarArgs>
	static MethodBind *bind_method(N p_method_name, M p_method, VarArgs... p_args);

	template <typename N, typename M, typename... VarArgs>
	static MethodBind *bind_static_method(StringName p_class, N p_method_name, M p_method, VarArgs... p_args);

	template <typename M>
	static MethodBind *bind_vararg_method(uint32_t p_flags, StringName p_name, M p_method, const MethodInfo &p_info = MethodInfo(), const std::vector<Variant> &p_default_args = std::vector<Variant>{}, bool p_return_nil_is_variant = true);

	static void add_property_group(const StringName &p_class, const String &p_name, const String &p_prefix);
	static void add_property_subgroup(const StringName &p_class, const String &p_name, const String &p_prefix);
	static void add_property(const StringName &p_class, const PropertyInfo &p_pinfo, const StringName &p_setter, const StringName &p_getter, int p_index = -1);
	static void add_signal(const StringName &p_class, const MethodInfo &p_signal);
	static void bind_integer_constant(const StringName &p_class_name, const StringName &p_enum_name, const StringName &p_constant_name, GDExtensionInt p_constant_value, bool p_is_bitfield = false);
	// Binds an implementation of a virtual method defined in Godot.
	static void bind_virtual_method(const StringName &p_class, const StringName &p_method, GDExtensionClassCallVirtual p_call);
	// Add a new virtual method that can be implemented by scripts.
	static void add_virtual_method(const StringName &p_class, const MethodInfo &p_method, const Vector<StringName> &p_arg_names = Vector<StringName>());

	static MethodBind *get_method(const StringName &p_class, const StringName &p_method);

	static GDExtensionClassCallVirtual get_virtual_func(void *p_userdata, GDExtensionConstStringNamePtr p_name);
	static const GDExtensionInstanceBindingCallbacks *get_instance_binding_callbacks(const StringName &p_class);

	static void initialize(GDExtensionInitializationLevel p_level);
	static void deinitialize(GDExtensionInitializationLevel p_level);

	CLASSDB_SINGLETON_FORWARD_METHODS;
};

#define BIND_CONSTANT(m_constant) \
	::godot::ClassDB::bind_integer_constant(get_class_static(), "", #m_constant, m_constant);

#define BIND_ENUM_CONSTANT(m_constant) \
	::godot::ClassDB::bind_integer_constant(get_class_static(), ::godot::_gde_constant_get_enum_name(m_constant, #m_constant), #m_constant, m_constant);

#define BIND_BITFIELD_FLAG(m_constant) \
	::godot::ClassDB::bind_integer_constant(get_class_static(), ::godot::_gde_constant_get_bitfield_name(m_constant, #m_constant), #m_constant, m_constant, true);

#define BIND_VIRTUAL_METHOD(m_class, m_method)                                                                                                \
	{                                                                                                                                         \
		auto _call##m_method = [](GDExtensionObjectPtr p_instance, const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr p_ret) -> void { \
			call_with_ptr_args(reinterpret_cast<m_class *>(p_instance), &m_class::m_method, p_args, p_ret);                                   \
		};                                                                                                                                    \
		::godot::ClassDB::bind_virtual_method(m_class::get_class_static(), #m_method, _call##m_method);                                       \
	}

template <typename T, bool is_abstract>
void ClassDB::_register_class(bool p_virtual, bool p_exposed, bool p_runtime) {
	static_assert(TypesAreSame<typename T::self_type, T>::value, "Class not declared properly, please use GDCLASS.");
	static_assert(!FunctionsAreSame<T::self_type::_bind_methods, T::parent_type::_bind_methods>::value, "Class must declare 'static void _bind_methods'.");
	static_assert(!std::is_abstract_v<T> || is_abstract, "Class is abstract, please use GDREGISTER_ABSTRACT_CLASS.");
	instance_binding_callbacks[T::get_class_static()] = &T::_gde_binding_callbacks;

	// Register this class within our plugin
	ClassInfo cl;
	cl.name = T::get_class_static();
	cl.parent_name = T::get_parent_class_static();
	cl.level = current_level;
	std::unordered_map<StringName, ClassInfo>::iterator parent_it = classes.find(cl.parent_name);
	if (parent_it != classes.end()) {
		// Assign parent if it is also a custom class
		cl.parent_ptr = &parent_it->second;
	}
	classes[cl.name] = cl;
	class_register_order.push_back(cl.name);

	// Register this class with Godot
	GDExtensionClassCreationInfo3 class_info = {
		p_virtual, // GDExtensionBool is_virtual;
		is_abstract, // GDExtensionBool is_abstract;
		p_exposed, // GDExtensionBool is_exposed;
		p_runtime, // GDExtensionBool is_runtime;
		T::set_bind, // GDExtensionClassSet set_func;
		T::get_bind, // GDExtensionClassGet get_func;
		T::has_get_property_list() ? T::get_property_list_bind : nullptr, // GDExtensionClassGetPropertyList get_property_list_func;
		T::free_property_list_bind, // GDExtensionClassFreePropertyList2 free_property_list_func;
		T::property_can_revert_bind, // GDExtensionClassPropertyCanRevert property_can_revert_func;
		T::property_get_revert_bind, // GDExtensionClassPropertyGetRevert property_get_revert_func;
		T::validate_property_bind, // GDExtensionClassValidateProperty validate_property_func;
		T::notification_bind, // GDExtensionClassNotification2 notification_func;
		T::to_string_bind, // GDExtensionClassToString to_string_func;
		nullptr, // GDExtensionClassReference reference_func;
		nullptr, // GDExtensionClassUnreference unreference_func;
		&_create_instance_func<T>, // GDExtensionClassCreateInstance create_instance_func; /* this one is mandatory */
		T::free, // GDExtensionClassFreeInstance free_instance_func; /* this one is mandatory */
		&_recreate_instance_func<T>, // GDExtensionClassRecreateInstance recreate_instance_func;
		&ClassDB::get_virtual_func, // GDExtensionClassGetVirtual get_virtual_func;
		nullptr, // GDExtensionClassGetVirtualCallData get_virtual_call_data_func;
		nullptr, // GDExtensionClassCallVirtualWithData call_virtual_func;
		nullptr, // GDExtensionClassGetRID get_rid;
		(void *)&T::get_class_static(), // void *class_userdata;
	};

	internal::gdextension_interface_classdb_register_extension_class3(internal::library, cl.name._native_ptr(), cl.parent_name._native_ptr(), &class_info);

	// call bind_methods etc. to register all members of the class
	T::initialize_class();

	// now register our class within ClassDB within Godot
	initialize_class(classes[cl.name]);
}

template <typename T>
void ClassDB::register_class(bool p_virtual) {
	ClassDB::_register_class<T, false>(p_virtual);
}

template <typename T>
void ClassDB::register_abstract_class() {
	ClassDB::_register_class<T, true>();
}

template <typename T>
void ClassDB::register_internal_class() {
	ClassDB::_register_class<T, false>(false, false);
}

template <typename T>
void ClassDB::register_runtime_class() {
	ClassDB::_register_class<T, false>(false, true, true);
}

template <typename N, typename M, typename... VarArgs>
MethodBind *ClassDB::bind_method(N p_method_name, M p_method, VarArgs... p_args) {
	Variant args[sizeof...(p_args) + 1] = { p_args..., Variant() }; // +1 makes sure zero sized arrays are also supported.
	const Variant *argptrs[sizeof...(p_args) + 1];
	for (uint32_t i = 0; i < sizeof...(p_args); i++) {
		argptrs[i] = &args[i];
	}
	MethodBind *bind = create_method_bind(p_method);
	return bind_methodfi(METHOD_FLAGS_DEFAULT, bind, p_method_name, sizeof...(p_args) == 0 ? nullptr : (const void **)argptrs, sizeof...(p_args));
}

template <typename N, typename M, typename... VarArgs>
MethodBind *ClassDB::bind_static_method(StringName p_class, N p_method_name, M p_method, VarArgs... p_args) {
	Variant args[sizeof...(p_args) + 1] = { p_args..., Variant() }; // +1 makes sure zero sized arrays are also supported.
	const Variant *argptrs[sizeof...(p_args) + 1];
	for (uint32_t i = 0; i < sizeof...(p_args); i++) {
		argptrs[i] = &args[i];
	}
	MethodBind *bind = create_static_method_bind(p_method);
	bind->set_instance_class(p_class);
	return bind_methodfi(0, bind, p_method_name, sizeof...(p_args) == 0 ? nullptr : (const void **)argptrs, sizeof...(p_args));
}

template <typename M>
MethodBind *ClassDB::bind_vararg_method(uint32_t p_flags, StringName p_name, M p_method, const MethodInfo &p_info, const std::vector<Variant> &p_default_args, bool p_return_nil_is_variant) {
	MethodBind *bind = create_vararg_method_bind(p_method, p_info, p_return_nil_is_variant);
	ERR_FAIL_NULL_V(bind, nullptr);

	bind->set_name(p_name);
	bind->set_default_arguments(p_default_args);

	StringName instance_type = bind->get_instance_class();

	std::unordered_map<StringName, ClassInfo>::iterator type_it = classes.find(instance_type);
	if (type_it == classes.end()) {
		memdelete(bind);
		ERR_FAIL_V_MSG(nullptr, String("Class '{0}' doesn't exist.").format(Array::make(instance_type)));
	}

	ClassInfo &type = type_it->second;

	if (type.method_map.find(p_name) != type.method_map.end()) {
		memdelete(bind);
		ERR_FAIL_V_MSG(nullptr, String("Binding duplicate method: {0}::{1}.").format(Array::make(instance_type, p_method)));
	}

	// register our method bind within our plugin
	type.method_map[p_name] = bind;

	// and register with godot
	bind_method_godot(type.name, bind);

	return bind;
}

#define GDREGISTER_CLASS(m_class) ::godot::ClassDB::register_class<m_class>();
#define GDREGISTER_VIRTUAL_CLASS(m_class) ::godot::ClassDB::register_class<m_class>(true);
#define GDREGISTER_ABSTRACT_CLASS(m_class) ::godot::ClassDB::register_abstract_class<m_class>();
#define GDREGISTER_INTERNAL_CLASS(m_class) ::godot::ClassDB::register_internal_class<m_class>();
#define GDREGISTER_RUNTIME_CLASS(m_class) ::godot::ClassDB::register_runtime_class<m_class>();

} // namespace godot

CLASSDB_SINGLETON_VARIANT_CAST;

#endif // GODOT_CLASS_DB_HPP

				   GNU LESSER GENERAL PUBLIC LICENSE
					   Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.


  This version of the GNU Lesser General Public License incorporates
the terms and conditions of version 3 of the GNU General Public
License, supplemented by the additional permissions listed below.

  0. Additional Definitions.

  As used herein, "this License" refers to version 3 of the GNU Lesser
General Public License, and the "GNU GPL" refers to version 3 of the GNU
General Public License.

  "The Library" refers to a covered work governed by this License,
other than an Application or a Combined Work as defined below.

  An "Application" is any work that makes use of an interface provided
by the Library, but which is not otherwise based on the Library.
Defining a subclass of a class defined by the Library is deemed a mode
of using an interface provided by the Library.

  A "Combined Work" is a work produced by combining or linking an
Application with the Library.  The particular version of the Library
with which the Combined Work was made is also called the "Linked
Version".

  The "Minimal Corresponding Source" for a Combined Work means the
Corresponding Source for the Combined Work, excluding any source code
for portions of the Combined Work that, considered in isolation, are
based on the Application, and not on the Linked Version.

  The "Corresponding Application Code" for a Combined Work means the
object code and/or source code for the Application, including any data
and utility programs needed for reproducing the Combined Work from the
Application, but excluding the System Libraries of the Combined Work.

  1. Exception to Section 3 of the GNU GPL.

  You may convey a covered work under sections 3 and 4 of this License
without being bound by section 3 of the GNU GPL.

  2. Conveying Modified Versions.

  If you modify a copy of the Library, and, in your modifications, a
facility refers to a function or data to be supplied by an Application
that uses the facility (other than as an argument passed when the
facility is invoked), then you may convey a copy of the modified
version:

   a) under this License, provided that you make a good faith effort to
   ensure that, in the event an Application does not supply the
   function or data, the facility still operates, and performs
   whatever part of its purpose remains meaningful, or

   b) under the GNU GPL, with none of the additional permissions of
   this License applicable to that copy.

  3. Object Code Incorporating Material from Library Header Files.

  The object code form of an Application may incorporate material from
a header file that is part of the Library.  You may convey such object
code under terms of your choice, provided that, if the incorporated
material is not limited to numerical parameters, data structure
layouts and accessors, or small macros, inline functions and templates
(ten or fewer lines in length), you do both of the following:

   a) Give prominent notice with each copy of the object code that the
   Library is used in it and that the Library and its use are
   covered by this License.

   b) Accompany the object code with a copy of the GNU GPL and this license
   document.

  4. Combined Works.

  You may convey a Combined Work under terms of your choice that,
taken together, effectively do not restrict modification of the
portions of the Library contained in the Combined Work and reverse
engineering for debugging such modifications, if you also do each of
the following:

   a) Give prominent notice with each copy of the Combined Work that
   the Library is used in it and that the Library and its use are
   covered by this License.

   b) Accompany the Combined Work with a copy of the GNU GPL and this license
   document.

   c) For a Combined Work that displays copyright notices during
   execution, include the copyright notice for the Library among
   these notices, as well as a reference directing the user to the
   copies of the GNU GPL and this license document.

   d) Do one of the following:

	   0) Convey the Minimal Corresponding Source under the terms of this
	   License, and the Corresponding Application Code in a form
	   suitable for, and under terms that permit, the user to
	   recombine or relink the Application with a modified version of
	   the Linked Version to produce a modified Combined Work, in the
	   manner specified by section 6 of the GNU GPL for conveying
	   Corresponding Source.

	   1) Use a suitable shared library mechanism for linking with the
	   Library.  A suitable mechanism is one that (a) uses at run time
	   a copy of the Library already present on the user's computer
	   system, and (b) will operate properly with a modified version
	   of the Library that is interface-compatible with the Linked
	   Version.

   e) Provide Installation Information, but only if you would otherwise
   be required to provide such information under section 6 of the
   GNU GPL, and only to the extent that such information is
   necessary to install and execute a modified version of the
   Combined Work produced by recombining or relinking the
   Application with a modified version of the Linked Version. (If
   you use option 4d0, the Installation Information must accompany
   the Minimal Corresponding Source and Corresponding Application
   Code. If you use option 4d1, you must provide the Installation
   Information in the manner specified by section 6 of the GNU GPL
   for conveying Corresponding Source.)

  5. Combined Libraries.

  You may place library facilities that are a work based on the
Library side by side in a single library together with other library
facilities that are not Applications and are not covered by this
License, and convey such a combined library under terms of your
choice, if you do both of the following:

   a) Accompany the combined library with a copy of the same work based
   on the Library, uncombined with any other library facilities,
   conveyed under the terms of this License.

   b) Give prominent notice with the combined library that part of it
   is a work based on the Library, and explaining where to find the
   accompanying uncombined form of the same work.

  6. Revised Versions of the GNU Lesser General Public License.

  The Free Software Foundation may publish revised and/or new versions
of the GNU Lesser General Public License from time to time. Such new
versions will be similar in spirit to the present version, but may
differ in detail to address new problems or concerns.

  Each version is given a distinguishing version number. If the
Library as you received it specifies that a certain numbered version
of the GNU Lesser General Public License "or any later version"
applies to it, you have the option of following the terms and
conditions either of that published version or of any later version
published by the Free Software Foundation. If the Library as you
received it does not specify a version number of the GNU Lesser
General Public License, you may choose any version of the GNU Lesser
General Public License ever published by the Free Software Foundation.

  If the Library as you received it specifies that a proxy can decide
whether future versions of the GNU Lesser General Public License shall
apply, that proxy's public statement of acceptance of any version is
permanent authorization for you to choose that version for the
Library.



extends Control

#region Onready Vars
@onready var songTitleLabel : Label = $MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/MarginContainer/Panel/HBoxContainer/HBoxContainer/VBoxContainer/SongTitle
@onready var songAuthorLabel : Label = $MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/MarginContainer/Panel/HBoxContainer/HBoxContainer/VBoxContainer/Author
@onready var currentDurationLabel : Label = $MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/MarginContainer/Panel/HBoxContainer/HBoxContainer/VBoxContainer/HBoxContainer/CurrentDuration
@onready var progressBar : HSlider = $MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/MarginContainer/Panel/HBoxContainer/HBoxContainer/VBoxContainer/HBoxContainer/ProgressSlider
@onready var totalDurationLabel : Label = $MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/MarginContainer/Panel/HBoxContainer/HBoxContainer/VBoxContainer/HBoxContainer/TotalDuration
@onready var volumeSlider : VSlider = $MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/MarginContainer/Panel/HBoxContainer/HBoxContainer/VBoxContainer/HBoxContainer/volumeButton/Panel/VolumeSlider
@onready var smooth_scroll_container: ScrollContainer = $MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/SmoothScrollContainer
@onready var songElementsContainer : VBoxContainer = $MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/SmoothScrollContainer/HBoxContainer/VBoxContainer

@onready var song_lyrics_label: Label = $"MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Song Info/HBoxContainer/HBoxContainer/Panel/VBoxContainer/PanelContainer/songLyricsLabel"
@onready var song_lyrics_http_request: HTTPRequest = $"MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Song Info/HBoxContainer/HBoxContainer/Panel/VBoxContainer/PanelContainer/songLyricsLabel/songLyricsHTTPRequest"

@onready var manual_search_popup_control: Control = $"MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Song Info/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer/manuallySearchLyricsButton/manualSearchPopupControl"

#endregion

var SongElementScene: PackedScene = preload("res://Resources/songElement.tscn")
var globalUserDataPath : String = OS.get_user_data_dir()

#region EQ Variables Region
@onready var EQBands : Array[VSlider] = [
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer/VBoxContainer/VBoxContainer/VSlider,
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer/VBoxContainer2/VBoxContainer/VSlider,
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer/VBoxContainer3/VBoxContainer/VSlider,
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer/VBoxContainer4/VBoxContainer/VSlider,
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer/VBoxContainer5/VBoxContainer/VSlider,
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer/VBoxContainer6/VBoxContainer/VSlider,
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer/VBoxContainer7/VBoxContainer/VSlider,
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer/VBoxContainer8/VBoxContainer/VSlider,
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer/VBoxContainer9/VBoxContainer/VSlider,
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer/VBoxContainer10/VBoxContainer/VSlider,
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer/VBoxContainer11/VBoxContainer/VSlider,
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer2/VBoxContainer/VBoxContainer/VSlider,
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer2/VBoxContainer2/VBoxContainer/VSlider,
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer2/VBoxContainer3/VBoxContainer/VSlider,
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer2/VBoxContainer4/VBoxContainer/VSlider,
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer2/VBoxContainer5/VBoxContainer/VSlider,
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer2/VBoxContainer6/VBoxContainer/VSlider,
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer2/VBoxContainer7/VBoxContainer/VSlider,
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer2/VBoxContainer8/VBoxContainer/VSlider,
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer2/VBoxContainer9/VBoxContainer/VSlider,
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Equalizer/HBoxContainer/HBoxContainer/Panel/VBoxContainer/HBoxContainer2/VBoxContainer10/VBoxContainer/VSlider
]

#region Presets
var defaultPreset : PackedFloat32Array = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
var bassBoostedPreset : PackedFloat32Array = [13.586, 16.237, 18.888, 13.586, 14.47, -0.552, 13.586, 14.47, -4.971, 2.982, 13.586, 9.074, 10.824, 3.827, -0.547, 8.2, -8.418, -21.538, -32.034, -39.905, -47.777]
var enhanchedVocalsPreset : PackedFloat32Array = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.952, 10.824, 7.325, 9.949, 13.586, 6.45, -1.421, -2.296, -0.552]
var powerfulPreset : PackedFloat32Array = [13.586, 16.237, 18.888, 13.586, 14.47, -0.552, 13.586, 14.47, 7.214, 3.5, 0.0, 0.0, 2.952, 10.824, 7.325, 9.949, 13.586, 6.45, -1.421, -2.296, -0.552]
var powerful2Preset : PackedFloat32Array = [13.586, 16.237, 18.888, 13.586, 14.47, -0.552, 13.586, 14.47, 7.214, 3.5, 0.0, -5.875, 4.198, 9.666, 16.572, 16.285, 13.586, 12.543, 12.256, 11.392, 7.363]
var powerful3Preset : PackedFloat32Array = [13.586, 16.237, 18.098, 20.112, 18.961, 18.385, 17.522, -1.759, 8.025, 10.615, 10.903, -5.875, 4.198, 9.666, 16.572, 16.285, 13.586, 12.543, 12.256, 11.392, 7.363]
var powerful4Preset : PackedFloat32Array = [13.586, 16.237, 18.098, 20.112, 18.961, 18.385, 17.522, -1.759, 8.025, 10.615, 10.903, 7.651, 12.543, 11.392, 11.392, 16.285, 14.408, 19.45, 20.221, 21.675, 21.752]
var customPreset1 : PackedFloat32Array = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#endregion

@onready var EQ21Effect : AudioEffectEQ21 = AudioServer.get_bus_effect(0, 1)
var EQbandDragStarted : bool = false

#endregion

var currentSongElement : Node

var location : String = ""

var fullscreen : bool = false
var playing : bool = false
var loop : bool = false

var progressBarDragStarted : bool = false

#region Volume Variables
var volumeSliderDragStarted : bool = false

var volumeButtonHover : bool = false
var volumeSliderHover : bool = false
var volumeSliderPanelHover : bool = false
#endregion

#region Window Management
var dragging : bool = false
var mouse_pos : Vector2 = Vector2.ZERO
var drag_from : Vector2 = Vector2.ZERO
#endregion

var authorNameToRequestImage: String = ""
var songTitleToRequestImage: String = ""

var downloadingTrackID: String = ""
var downloadingSongAuthorName: String = ""
var downloadingSongName: String = ""

var savify_output: Array = []

var lyricsFullscreen: bool = false

var importerSongsThread: Thread


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	check_songs_dir_exists()

	for i: int in EQBands.size(): # EQ Part, sets the index to each band slider (0, 1, 2 etc.)
		EQBands[i].setEQNumber(i)

	setAudioBusVolume(-16)
	volumeSlider.value = -16

	if location == null or location == "":
		print("Location is void!")
	else:
		print("Location isn't void!")

	$"MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Song Info/HBoxContainer/HBoxContainer/Panel/VBoxContainer/PanelContainer/songLyricsLabel".mouse_filter = MOUSE_FILTER_IGNORE
	%loadingLyricsCenterContainer.mouse_filter = MOUSE_FILTER_IGNORE
	$"MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Song Info/HBoxContainer/HBoxContainer/Panel/VBoxContainer/PanelContainer/songLyricsLabel/loadingLyricsCenterContainer/MarginContainer".mouse_filter = MOUSE_FILTER_IGNORE
	%loadingLyricsLabel.mouse_filter = MOUSE_FILTER_IGNORE
	%loadingLyricsBytesLabel.mouse_filter = MOUSE_FILTER_IGNORE
	%loadingLyricsProgressBar.mouse_filter = MOUSE_FILTER_IGNORE


func check_songs_dir_exists() -> void:
	if !DirAccess.dir_exists_absolute(OS.get_system_dir(OS.SYSTEM_DIR_MUSIC).path_join("GAMP-Downloaded")):
		DirAccess.make_dir_absolute(OS.get_system_dir(OS.SYSTEM_DIR_MUSIC).path_join("GAMP-Downloaded"))


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta: float) -> void:
	if %MusicPlayer.playing:
		progressBar.value = %MusicPlayer.get_playback_position()
		currentDurationLabel.text = formatSongDuration(%MusicPlayer.get_playback_position())
		currentSongElement.get_node("Panel/MarginContainer/HBoxContainer/HBoxContainer/VBoxContainer/HBoxContainer/songProgressBar").value = progressBar.value
		currentSongElement.get_node("Panel/MarginContainer/HBoxContainer/HBoxContainer/VBoxContainer/HBoxContainer/CurrentDuration").text = formatSongDuration($MusicPlayer.get_playback_position())
		currentSongElement.currentSongTimestamp = %MusicPlayer.get_playback_position()


func _unhandled_key_input(event: InputEvent) -> void:
	if event.is_action_pressed("play_pause_key"):
		pauseAndResume()
	elif event.is_action_pressed("stop_key"):
		stop()
	elif event.is_action_pressed("prev_key"):
		prev()
	elif event.is_action_pressed("next_key"):
		next()


#region Window Managament Section
func _on_minimize_button_pressed() -> void:
	DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_MINIMIZED)


func _on_maximize_button_pressed() -> void:
	if !fullscreen:
		DisplayServer.window_set_flag(DisplayServer.WINDOW_FLAG_BORDERLESS, false)
		DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_MAXIMIZED)

		$MarginContainer["theme_override_constants/margin_left"] = 0
		$MarginContainer["theme_override_constants/margin_top"] = 0
		$MarginContainer["theme_override_constants/margin_right"] = 0
		$MarginContainer["theme_override_constants/margin_bottom"] = 0

		%loadingLyricsLabel.label_settings.font_size = 22
		%songLyricsLabel.label_settings.font_size = 32

		fullscreen = true
		DisplayServer.window_set_flag(DisplayServer.WINDOW_FLAG_BORDERLESS, true)
	else:
		DisplayServer.window_set_flag(DisplayServer.WINDOW_FLAG_BORDERLESS, true)
		DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_WINDOWED)


		$MarginContainer["theme_override_constants/margin_left"] = 4
		$MarginContainer["theme_override_constants/margin_top"] = 4
		$MarginContainer["theme_override_constants/margin_right"] = 4
		$MarginContainer["theme_override_constants/margin_bottom"] = 4

		%loadingLyricsLabel.label_settings.font_size = 16
		%songLyricsLabel.label_settings.font_size = 26

		fullscreen = false


func _on_close_button_pressed() -> void:
	get_tree().quit()


func _on_panel_gui_input(event: InputEvent) -> void:
	if !fullscreen:
		if event is InputEventMouseButton:
			if event.button_index == MOUSE_BUTTON_LEFT:
				if event.pressed and !dragging:
					#DisplayServer.cursor_set_shape(DisplayServer.CURSOR_DRAG)
					$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer2.mouse_default_cursor_shape = CURSOR_DRAG
					dragging = true
					drag_from = get_global_mouse_position()
				else:
					#DisplayServer.cursor_set_shape(DisplayServer.CURSOR_ARROW)
					$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer2.mouse_default_cursor_shape = CURSOR_ARROW
					dragging = false
		if event is InputEventMouseMotion:
			if dragging:
				#DisplayServer.cursor_set_shape(DisplayServer.CURSOR_CAN_DROP)
				var window: Window = self.get_window()
				var real_mouse_pos: Vector2 = get_global_mouse_position() - drag_from
				@warning_ignore("narrowing_conversion")
				window.position += Vector2i(real_mouse_pos.x, real_mouse_pos.y)
	else:
		if event is InputEventMouseButton:
			if event.button_index == MOUSE_BUTTON_LEFT:
				if event.pressed:
					_on_maximize_button_pressed()
					dragging = true
#endregion


#region Importing Section
func _on_import_dir_button_pressed() -> void:
	check_songs_dir_exists()
	var filters: PackedStringArray = ["*.mp3", "*.wav", "*.ogg"]
	DisplayServer.file_dialog_show("Import Directory", OS.get_system_dir(OS.SYSTEM_DIR_MUSIC).path_join("GAMP-Downloaded"), "", true, DisplayServer.FILE_DIALOG_MODE_OPEN_DIR, filters, onNativeFileDialogDirSelected)


func onNativeFileDialogDirSelected(status: bool, selected_paths: PackedStringArray, selected_filter_index: int) -> void:
	if status:
		printt(status, selected_paths, selected_filter_index)

		dirSelectedImportSong(selected_paths[0])
	else:
		print("Failed to import folder")


func hideEmptyListSongLabel() -> void:
	%emptySongListLabel.hide()


func dirSelectedImportSong(dir: String) -> void:
	var diraccess: DirAccess = DirAccess.open(dir)
	var filesInDir: PackedStringArray = diraccess.get_files()

	if filesInDir.size() != 0:
		call_deferred("hideEmptyListSongLabel")

		importerSongsThread = Thread.new()
		importerSongsThread.start(importSongsThreaded.bind(filesInDir, dir))


func importSongsThreaded(filesInDir: PackedStringArray, dir: String) -> void:
	for i: int in filesInDir.size():
		var fileName: String = filesInDir[i]
		var fileExtension: String = fileName.get_extension()
		var baseFileName: String = fileName.get_basename() # File name, without the extension, ex. "g.mp3" becomes "g"
		var splittedFileName: PackedStringArray = baseFileName.split("")
			# now we gotta take everything before the "-" so that it becomes the Author and whatever it's over "-" it's the Song Title
		var delimiterFound : bool = false

		var songAuthor: String = ""
		var songTitle: String = ""

		for j: int in splittedFileName.size():
			var fileNameChars: String = splittedFileName[j]
			if !delimiterFound:
				if fileNameChars != "-":
					songAuthor += fileNameChars
					delimiterFound = false
				else:
					delimiterFound = true
			else:
				songTitle += fileNameChars

		songAuthor = songAuthor.strip_edges(true, true)
		songTitle = songTitle.strip_edges(true, true)

			#print("Done reading the song: '", songAuthor, " - ", songTitle, "', importing it now...")

		if fileExtension == "wav" or fileExtension == "mp3" or fileExtension == "ogg":
			var SongElement: MarginContainer = SongElementScene.instantiate()
			songElementsContainer.call_deferred("add_child", SongElement, true) #.add_child(SongElement)

				#SongElement.setTitle(songTitle)
				#SongElement.setAuthor(songAuthor)
			SongElement.setSongFileName(fileName)
			SongElement.setSongFileNamePath(dir.path_join(fileName))
			SongElement.setSongFileNameDir(dir)
			SongElement.setCurrentDuration("0:00")

			var fileCompletePath: String = dir.path_join(fileName)
			var songTotalDuration: String

			match fileExtension:
				"wav":
					var tempStream: AudioStreamWAV = AudioStreamWAV.new()

					tempStream.set_format(AudioStreamWAV.FORMAT_16_BITS)
					tempStream.mix_rate = 48000
					tempStream.stereo = true
					tempStream.data = load_song_data(fileCompletePath)

					songTotalDuration = formatSongDuration(tempStream.get_length())

				"mp3":
					var tempStream: AudioStreamMP3 = AudioStreamMP3.new()

					tempStream.data = load_song_data(fileCompletePath)
					songTotalDuration = formatSongDuration(tempStream.get_length())

				"ogg":
					var tempStream: AudioStreamOggVorbis = AudioStreamOggVorbis.load_from_file(fileCompletePath)
					songTotalDuration = formatSongDuration(tempStream.get_length())


			SongElement.call_deferred("setTotalDuration", songTotalDuration)

		else:
			print("File: '", fileName, "' is not an .mp3/.wav/.ogg, skipping...")

		print(filesInDir[i])

	call_deferred("importer_Thread_Finished_Importing")


func importer_Thread_Finished_Importing() -> void:
	importerSongsThread.wait_to_finish()


func importSingleSong(filePath: String) -> void:
	var fileName: String = filePath.get_basename() + "." + filePath.get_extension()
	var fileExtension: String = filePath.get_extension()
	var baseFileName: String = filePath.get_basename() # File name, without the extention, ex. "g.mp3" becomes "g"

			#region Maybe Delete
			#var songAuthor : String = ""
			#var songTitle : String = ""

			#for j: int in splittedFileName.size():
			#	var fileNameChars : String = splittedFileName[j]
			#	if !delimiterFound:
			#		if fileNameChars != "-":
			#			songAuthor += fileNameChars
			#			delimiterFound = false
			#		else:
			#			delimiterFound = true
			#	else:
			#		songTitle += fileNameChars

			#songAuthor = songAuthor.strip_edges(true, true)
			#songTitle = songTitle.strip_edges(true, true)
			#endregion

	print("Done.")

	if fileExtension == "wav" or fileExtension == "mp3" or fileExtension == "ogg":
		var SongElement: MarginContainer = SongElementScene.instantiate()
		songElementsContainer.call_deferred("add_child", SongElement) #.add_child(SongElement)

				#SongElement.setTitle(songTitle)
				#SongElement.setAuthor(songAuthor)
		SongElement.setSongFileName(fileName)

		SongElement.setSongFileNamePath(filePath)
		SongElement.setSongFileNameDir(fileName.get_base_dir())

		SongElement.setCurrentDuration("0:00")

		var fileCompletePath: String = fileName
		var songTotalDuration: String

		match fileExtension:
			"wav":
				var tempStream: AudioStreamWAV = AudioStreamWAV.new()

				tempStream.set_format(AudioStreamWAV.FORMAT_16_BITS)
				tempStream.mix_rate = 48000
				tempStream.stereo = true
				tempStream.data = load_song_data(fileCompletePath)

				songTotalDuration = formatSongDuration(tempStream.get_length())

			"mp3":
				var tempStream: AudioStreamMP3 = AudioStreamMP3.new()

				tempStream.data = load_song_data(fileCompletePath)
				songTotalDuration = formatSongDuration(tempStream.get_length())

			"ogg":
				var tempStream : AudioStreamOggVorbis = AudioStreamOggVorbis.load_from_file(fileCompletePath)
				songTotalDuration = formatSongDuration(tempStream.get_length())

		SongElement.call_deferred("setTotalDuration", songTotalDuration)
	else:
		print("File: '", fileName, "' is not an .mp3/.wav/.ogg, skipping...")


func load_song_data(path: String) -> PackedByteArray:
	var file: FileAccess = FileAccess.open(path, FileAccess.READ)

	return file.get_buffer(file.get_length())

#endregion


func _on_play_button_pressed() -> void:
	pauseAndResume()


func prev() -> void: # goes to the previous song (if there is one)
	if %MusicPlayer.stream != null:
		var parentChildCount: int = songElementsContainer.get_child_count()
		var currentSongElementPosition: int = currentSongElement.get_index()
		var prevChildIndex: int = currentSongElementPosition - 1
		if prevChildIndex > 1:  # >2 ??
				#region  I prefer to use a signal or use a group, nodes are not called is slow and confusing
			var nextChild: MarginContainer = songElementsContainer.get_child(prevChildIndex)
			nextChild._on_song_element_button_pressed()
		else:
			var nextChild: MarginContainer = songElementsContainer.get_child(parentChildCount - 1)
			nextChild._on_song_element_button_pressed()
			#region end #   #nextChild._on_song_element_button_pressed()

func next() -> void: # skips to the next song (if there is one, if there is not, it goes to the first one)
	if % MusicPlayer.stream != null: #
		var parentChildCount: int = songElementsContainer.get_child_count()
		var currentSongElementPosition: int = currentSongElement.get_index()
		var nextChildIndex: int = currentSongElementPosition + 1
		if nextChildIndex < parentChildCount:
			var nextChild: MarginContainer = songElementsContainer.get_child(nextChildIndex)
		#region  I prefer to use a signal or use a group, nodes are not called is slow and confusing
			nextChild._on_song_element_button_pressed()

		else:
			var nextChild: MarginContainer = songElementsContainer.get_child(2)

			nextChild._on_song_element_button_pressed()

			#region end #   # nextChild._on_song_element_button_pressed()


func pauseAndResume() -> void: # pauses if it's playing and resumes if it's not (but has started) a song
	if %MusicPlayer.stream_paused == true:

		%playButton.add_theme_font_size_override("font_size", 22)
		%playButton.text = "ıı"

		%MusicPlayer.stream_paused = false
	else:

		%playButton.add_theme_font_size_override("font_size", 16)
		%playButton.text = "▶"

		%MusicPlayer.stream_paused = true


func formatSongDuration(duration: float) -> String:
	var minutes : int = int(duration / 60.0)
	var seconds : int = int(duration - minutes * 60.0)
	var finalDuration : String = str(minutes).pad_zeros(1) + ":" + str(seconds).pad_zeros(2)
	return finalDuration


func reverseFormatSongDuration(duration: String) -> float:
	var parts: PackedStringArray = duration.split(":")
	if parts.size() != 2:
		push_error("Formato durata non valido. Deve essere in formato 'mm:ss'")
		return 0.0

	var minutes: int = int(parts[0])
	var seconds: int = int(parts[1])

	return float(minutes * 60.0 + seconds)


func setSongTitleAuthorDuration(author : String, title : String, _duration : String) -> void:
	songAuthorLabel.text = author
	songTitleLabel.text = title
	totalDurationLabel.text = formatSongDuration(%MusicPlayer.stream.get_length())

	# from here it changes the progressbar's values to match total duration etc.,
	# i could use some math but why should i make the cpu calculate things when
	# i can just change existing values?
	progressBar.max_value = %MusicPlayer.stream.get_length()
	currentSongElement.get_node("Panel/MarginContainer/HBoxContainer/HBoxContainer/VBoxContainer/HBoxContainer/songProgressBar").max_value = progressBar.max_value


func play(stream: AudioStream = null, fromPosition: float = 0.0) -> void: # plays the song
	if stream != null:
		%MusicPlayer.stream = stream
	%MusicPlayer.play(fromPosition)


## Resets the song to 0:00 and pauses it
func stop() -> void:
	%MusicPlayer.play(0.0)

	%playButton.add_theme_font_size_override("font_size", 16)
	%playButton.text = "▶"

	%MusicPlayer.stream_paused = true


func loopOn() -> void:
	if !%MusicPlayer.stream == AudioStreamWAV:
		%MusicPlayer.stream.loop = true


func loopOff() -> void:
	if !%MusicPlayer.stream == AudioStreamWAV:
		%MusicPlayer.stream.loop = false


func loopButton() -> void: # loops the song
	if !loop:
		loop = true
	else:
		loop = false


func loadSongFile(filepath : String) -> AudioStream: # loads the song file (supports only .wav, .mp3 and .ogg)
	var extension: String = filepath.get_extension()
	
	var data: PackedByteArray = load_song_data(filepath)

	match extension:
		"mp3":
			print("file is an mp3!")

			var newStream : AudioStreamMP3 = AudioStreamMP3.new()
			newStream.set_data(data)
			print()

			data.clear()
			return newStream
		"wav":
			print("file is a wav!")
			var newStream : AudioStreamWAV = AudioStreamWAV.new()
			newStream.set_format(AudioStreamWAV.FORMAT_16_BITS)
			newStream.mix_rate = 48000
			newStream.stereo = true
			newStream.set_data(data)

			data.clear()
			return newStream
		"ogg":
			print("file is an ogg!")
			var newStream : AudioStreamOggVorbis = AudioStreamOggVorbis.load_from_file(filepath)

			data.clear()
			return newStream
		_:
			return null


func songElementSelectedFunction(songElementNode : Node, songFileName : String, songFileNamePath : String, songFileNameDir : String, songAuthor : String, songTitle : String, songTotalDuration : String, songCurrentTimestamp: float) -> void:
	if currentSongElement != null and currentSongElement.playing :
		currentSongElement.stopPlayingAnimation()
		currentSongElement.playing = false
	songElementNode.playPlayingAnimation()

	print("\nsongElementNode is: " + songElementNode.name)
	print("\nsongFileName is: " + songFileName)
	print("\nsongFileNamePath is: " + songFileNamePath)
	print("\nsongFileNameDir is: " + songFileNameDir)

	%playButton.add_theme_font_size_override("font_size", 20)
	%playButton.text = "ıı"

	currentSongElement = songElementNode

	currentSongElement.playing = true

	play(loadSongFile(songFileNamePath), songCurrentTimestamp)
	setSongTitleAuthorDuration(songAuthor, songTitle, songTotalDuration)

	# Lyrics Retriever Section:
	if currentSongElement.song_thumbnail_texture_rect.texture is not ImageTexture:
		$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer.current_tab = 2
		return
	requestSongLyrics(songTitle, songAuthor, songTotalDuration)
	#requestSongImage(songAuthor, songTitle)
	requestAuthorImage(songAuthor, songTitle)
	changeSongCoverImage(currentSongElement.song_thumbnail_texture_rect.texture)
	changeBGImage(currentSongElement.song_thumbnail_texture_rect.texture)

	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer.current_tab = 2


func changeBGImage(newBGImage: ImageTexture) -> void:
	var changeBGImageTween: Tween = create_tween()

	changeBGImageTween.set_ease(Tween.EASE_IN_OUT)
	changeBGImageTween.set_process_mode(Tween.TWEEN_PROCESS_IDLE)
	changeBGImageTween.set_trans(Tween.TRANS_QUAD)

	changeBGImageTween.tween_property(%BGTextureRect, "self_modulate", Color.BLACK, 0.2)
	changeBGImageTween.chain().tween_property(%BGTextureRect, "texture", newBGImage, 0)
	changeBGImageTween.chain().tween_property(%BGTextureRect, "self_modulate", Color.WHITE, 0.2)



func changeAuthorImage() -> void:
	var changeAuthorImageTween: Tween = create_tween()

	changeAuthorImageTween.set_ease(Tween.EASE_IN_OUT)
	changeAuthorImageTween.set_process_mode(Tween.TWEEN_PROCESS_IDLE)
	changeAuthorImageTween.set_trans(Tween.TRANS_QUAD)

	changeAuthorImageTween.tween_property(%authorCoverTextureRect, "self_modulate", Color.BLACK, 0.2)
	changeAuthorImageTween.chain().tween_callback(%authorCoverTextureRect.request) #(%songCoverTextureRect, "texture", newSongCoverImage, 0)
	changeAuthorImageTween.chain().tween_property(%authorCoverTextureRect, "self_modulate", Color.WHITE, 0.2).set_delay(0.2)


func changeSongCoverImage(newSongCoverImage: ImageTexture) -> void:
	var changeSongCoverImageTween : Tween = create_tween()

	changeSongCoverImageTween.set_ease(Tween.EASE_IN_OUT)
	changeSongCoverImageTween.set_process_mode(Tween.TWEEN_PROCESS_IDLE)
	changeSongCoverImageTween.set_trans(Tween.TRANS_QUAD)

	changeSongCoverImageTween.tween_property(%songCoverTextureRect, "self_modulate", Color.BLACK, 0.2)
	changeSongCoverImageTween.chain().tween_callback(%songCoverTextureRect.set_texture.bind(newSongCoverImage)) #(%songCoverTextureRect, "texture", newSongCoverImage, 0)
	changeSongCoverImageTween.chain().tween_property(%songCoverTextureRect, "self_modulate", Color.WHITE, 0.2)


func time_to_seconds(time_string: String) -> int:
	var parts: PackedStringArray = time_string.split(":")
	if parts.size() != 2:
		push_error("Formato tempo non valido. Deve essere in formato 'mm:ss'")
		return 0

	var minutes: int = int(parts[0])
	var seconds: int = int(parts[1])

	return minutes * 60 + seconds


#region Lyrics Management
func requestSongLyrics(Title: String, Author: String, Duration: String) -> void:
	## This is the API we get the Lyrics from
	var url_starting_part: String = "https://lrclib.net/api/get?artist_name="

	var songTitleURL: String = Title.uri_encode()

	var songAuthorURL: String = Author.uri_encode()

	var final_url: String = url_starting_part + songAuthorURL + "&track_name=" + songTitleURL + "&duration=" + str(time_to_seconds(Duration))

	print("Searching lyrics with URL: ", final_url)

	var headers: PackedStringArray = [
		"User-Agent: GAMP v1.0 (https://github.com/xShader1374/GAMP)"
	]

	LyricsSynchronizer.set_process(false)
	LyricsSynchronizer.lineCounter = 0
	LyricsSynchronizer.syncSeconds.clear()
	LyricsSynchronizer.lyricsLineLabels.clear()

	song_lyrics_http_request.cancel_request()
	song_lyrics_http_request.request(final_url, headers, HTTPClient.METHOD_GET)


	song_lyrics_label.text = ""

	%songLyricsLinesVBoxContainer.hide()
	%loadingLyricsCenterContainer.show()


func _on_song_lyrics_http_request_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray) -> void:
	# stop and hide the loading thingy

	printt(result, response_code, headers, body)
	if body.is_empty():
		prints("connection failure or a null value")
		return
		
	var parsedBody: Dictionary = {}
	parsedBody = JSON.parse_string(body.get_string_from_utf8())

	%loadingLyricsCenterContainer.hide()

	if response_code == 200:
		# Checks if body is not null
		if body:
			song_lyrics_label.text = ""

			if !parsedBody["instrumental"]:
				%songLyricsLinesVBoxContainer.show()
				%songLyricsLinesVBoxContainer.get_parent().scroll_to_top()

				loadImportedSyncedLyrics(parsedBody.syncedLyrics)
				LyricsSynchronizer.set_process(true)
			else:
				%loadingLyricsCenterContainer.hide()
				song_lyrics_label.text = "♪ Instrumental ♪"
	else:
		print(parsedBody)
		song_lyrics_label.text = "Can't find lyrics for this song.\n(Using LRCLIB API)"


func stripEverythingBetweenSomethingFromString(string : String, symbol1 : String, symbol2 : String) -> String:
	var newString : String = ""
	var insideDelimeters : bool = false

	for Char: String in string:

		if Char == symbol1:
			insideDelimeters = true
		elif Char == symbol2:
			insideDelimeters = false
		elif !insideDelimeters:
			newString += Char

	return newString


func stripSyncSecondsFromLyrics(FullLyrics : String) -> String:
	return stripEverythingBetweenSomethingFromString(FullLyrics, "[", "]")


func loadImportedSyncedLyrics(fullLyrics : String) -> void:
	var lyricsLinesOLD : Array[Node] = %songLyricsLinesVBoxContainer.get_children()

	for child: Label in lyricsLinesOLD:
		child.queue_free()

	var lyricsLines : PackedStringArray = fullLyrics.split("\n")

	# 10 characters

	for line: String in lyricsLines:

		var lyricLineStripped: String = line.right(line.length() - 10)

		LyricsSynchronizer.syncSeconds.append(line.left(10))


		var newLyricsLineNode: Label = Label.new()

		newLyricsLineNode.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		newLyricsLineNode.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
		newLyricsLineNode.label_settings = %loadingLyricsLabel.label_settings

		if lyricLineStripped == " ":
			newLyricsLineNode.text = "♪"
		else:
			newLyricsLineNode.text = lyricLineStripped.right(-1)

		newLyricsLineNode.pivot_offset = newLyricsLineNode.size / 2.0

		%songLyricsLinesVBoxContainer.add_child(newLyricsLineNode)

		LyricsSynchronizer.lyricsLineLabels.append(newLyricsLineNode)

		newLyricsLineNode.gui_input.connect(LyricsSynchronizer.lyrics_line_GUI_input_event.bind(newLyricsLineNode))
		newLyricsLineNode.mouse_entered.connect(LyricsSynchronizer.lyrics_line_mouse_entered.bind(newLyricsLineNode))
		newLyricsLineNode.mouse_exited.connect(LyricsSynchronizer.lyrics_line_mouse_exited.bind(newLyricsLineNode))

		newLyricsLineNode.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		newLyricsLineNode.mouse_default_cursor_shape = Control.CURSOR_POINTING_HAND
		newLyricsLineNode.mouse_filter = Control.MOUSE_FILTER_PASS

	print(LyricsSynchronizer.syncSeconds, LyricsSynchronizer.syncSeconds.size())
	print(LyricsSynchronizer.lyricsLineLabels, LyricsSynchronizer.lyricsLineLabels.size())
#endregion


func requestAuthorImage(Author: String, Title: String) -> void:
	authorNameToRequestImage = Author
	songTitleToRequestImage = Title
	%authorCoverTokenHTTPRequest.cancel_request()
	%authorCoverTokenHTTPRequest.request("https://open.spotify.com/get_access_token?reason=transport&productType=web_player")

	printt("Requesting image with params: ", Author, Title)


func _on_author_cover_token_http_request_request_completed(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	if response_code == 200:
		var parsedBody: Dictionary = JSON.parse_string(body.get_string_from_ascii())
		print(parsedBody)
		print("Token is: ", parsedBody["accessToken"])

		var finalAuthorName: String = authorNameToRequestImage.uri_encode()
		var finalTrackName: String = songTitleToRequestImage.uri_encode()

		%authorCoverHTTPRequest.cancel_request()
		%authorCoverHTTPRequest.request(("https://api.spotify.com/v1/search?type=artist&q=" + finalAuthorName + "&track=" + finalTrackName + "&decorate_restrictions=false&best_match=true&include_external=audio&limit=3"), ["Authorization: Bearer " + parsedBody["accessToken"]])
	else:
		printerr("Couldn't get TOKEN Author Image from Spotify API: ", response_code)


func _on_author_cover_http_request_request_completed(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	var parsedBody: Dictionary = JSON.parse_string(body.get_string_from_ascii()) if body.size() > 15 else {"response_code": response_code}
	var images: Array
	if response_code == 200:
		if typeof(parsedBody["artists"]["items"]) == TYPE_ARRAY and typeof(parsedBody["artists"]["items"][0]) == TYPE_ARRAY:

			images = parsedBody["artists"]["items"][0]["images"]
	else :
		push_error(" error de imagen 0 no es un array ")

	if !images.size():
		if typeof(parsedBody["artists"]["items"]) == TYPE_ARRAY and parsedBody["artists"]["items"].size() > 1:
			var second_item = parsedBody["artists"]["items"][1]
			if typeof(second_item.get("images", [])) == TYPE_ARRAY:

				images = parsedBody["artists"]["items"][1]["images"] 
		else :
			push_error(" error de imagen no es un array ")



		var foundURL: String = ""

		foundURL = images[0]["url"]

		%authorCoverTextureRect.url = foundURL
		changeAuthorImage()

		print("\nBest Match Author Image pfp URL: ", foundURL)
	else:
		%authorCoverTextureRect.url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fstatic.vecteezy.com%2Fsystem%2Fresources%2Fpreviews%2F010%2F892%2F324%2Foriginal%2Fx-transparent-free-png.png&f=1&nofb=1&ipt=a964354a8dad0bca3ff81c5fbf84ca99eb4420b3ed26f6b9b29e3dcb7a36232c&ipo=images"
		changeAuthorImage()
		printerr("Getting Author Cover: Something went wrong, code: ", response_code)
		print(parsedBody)


func requestSongImage(Author : String, Title : String) -> void:
	var output : Array = []
	var url_starting_part : String = "https://open.spotify.com/oembed?url="

	#OS.execute("C:/Users/shader/AppData/Roaming/Godot/app_userdata/GAMP/linkSongGetter.exe", [str("-a " + Author), str("-s " + Title)], output)

	var args : PackedStringArray = [
		str(globalUserDataPath + "/SpotifyLinkSongGetter.exe"),
		str("-a ", Author),
		str("-s ", Title)
	]

	print("Does link.txt exist? ", FileAccess.file_exists(globalUserDataPath + "/link.txt"))

	#OS.execute(globalUserDataPath + "/SpotifyLinkSongGetter.exe", args, output, true, true)

	OS.execute('powershell.exe', args, output, true, true)

	for i : int in output.size():
		print("\n--- ", str(output[i]), " ---\n")

	print("Does link.txt exist? ", FileAccess.file_exists(globalUserDataPath + "/link.txt"))



	var foundURL : String = FileAccess.get_file_as_string(globalUserDataPath + "/link.txt")

	#DirAccess.remove_absolute(globalUserDataPath + "/link.txt")

	print("SpotifyLinkSongGetter.exe found this link: " + foundURL)

	var final_url : String = url_starting_part + foundURL

	final_url = final_url.strip_edges(false, true)

	print("Searching song cover with URL: ", final_url)

	%songCoverHTTPRequest.cancel_request()
	var error : Error = %songCoverHTTPRequest.request(final_url)

	if error != OK:
		print("Couldn't Get Song Thumbnail Link: ", error)



func EQBandSliderDragStarted() -> void:
	EQbandDragStarted = true


func EQBandSliderDragEnded(_value_changed: bool) -> void:
	EQbandDragStarted = false


func EQBandSliderValueChanged(body: Node, value: float) -> void:
	#if EQbandDragStarted:
	EQ21Effect.set_band_gain_db(body.EQNumber, value)


func _on_progress_bar_2_drag_started() -> void:
	progressBarDragStarted = true


func _on_progress_bar_2_value_changed(value: float) -> void:
	if songElementsContainer.get_child_count() > 1 and progressBarDragStarted:

		if %MusicPlayer.stream_paused:
			%MusicPlayer.stream_paused = false
			%playButton.add_theme_font_size_override("font_size", 20)
			%playButton.text = "ıı"

		%MusicPlayer.seek(progressBar.value)


func _on_progress_bar_2_drag_ended(_value_changed: bool) -> void:
	progressBarDragStarted = false


func _on_volume_button_mouse_entered() -> void:
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/MarginContainer/Panel/HBoxContainer/HBoxContainer/VBoxContainer/HBoxContainer/volumeButton/AnimationPlayer.play("show")
	volumeButtonHover = true


func _on_volume_button_mouse_exited() -> void:
	volumeButtonHover = false


func _on_panel_mouse_exited() -> void:
	volumeSliderPanelHover = false

	if volumeSliderHover == false and volumeSliderPanelHover == false and volumeButtonHover == false:
		$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/MarginContainer/Panel/HBoxContainer/HBoxContainer/VBoxContainer/HBoxContainer/volumeButton/AnimationPlayer.play_backwards("show")


func _on_panel_mouse_entered() -> void:
	volumeSliderPanelHover = true


func _on_volume_slider_mouse_entered() -> void:
	volumeSliderHover = true


func _on_volume_slider_mouse_exited() -> void:
	volumeSliderHover = false


func setAudioBusVolume(volume : float) -> void:
	AudioServer.set_bus_volume_db(0, volume)


func _on_volume_button_toggled(toggled_on: bool) -> void:
	if toggled_on:
		AudioServer.set_bus_mute(0, true)
		$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/MarginContainer/Panel/HBoxContainer/HBoxContainer/VBoxContainer/HBoxContainer/volumeButton/volumeButton/Label.show()
	else:
		AudioServer.set_bus_mute(0, false)
		$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/MarginContainer/Panel/HBoxContainer/HBoxContainer/VBoxContainer/HBoxContainer/volumeButton/volumeButton/Label.hide()


func _on_volume_slider_drag_started() -> void:
	volumeSliderDragStarted = true


func _on_volume_slider_drag_ended(_value_changed: bool) -> void:
	volumeSliderDragStarted = false


func _on_volume_slider_value_changed(value: float) -> void:
	if volumeSliderDragStarted:
		setAudioBusVolume(value)


func _on_total_duration_gui_input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
		if event.pressed:
			print("premuto")
		#else:
		#	print("rilasciato")
	#TODO: make a "-2:39" or 4:20, essentially adding the option to view the remaining time or the total duration


func _on_presets_option_button_item_selected(index: int) -> void:
	match index:
		0:
			for i: int in EQ21Effect.get_band_count():
				EQ21Effect.set_band_gain_db(i, defaultPreset[i])
				EQBands[i].value = EQ21Effect.get_band_gain_db(i)
		1:
			for i: int in EQ21Effect.get_band_count():
				EQ21Effect.set_band_gain_db(i, bassBoostedPreset[i])
				EQBands[i].value = EQ21Effect.get_band_gain_db(i)
		2:
			for i: int in EQ21Effect.get_band_count():
				EQ21Effect.set_band_gain_db(i, enhanchedVocalsPreset[i])
				EQBands[i].value = EQ21Effect.get_band_gain_db(i)
		3:
			for i: int in EQ21Effect.get_band_count():
				EQ21Effect.set_band_gain_db(i, powerfulPreset[i])
				EQBands[i].value = EQ21Effect.get_band_gain_db(i)
		4:
			for i: int in EQ21Effect.get_band_count():
				EQ21Effect.set_band_gain_db(i, powerful2Preset[i])
				EQBands[i].value = EQ21Effect.get_band_gain_db(i)
		5:
			for i: int in EQ21Effect.get_band_count():
				EQ21Effect.set_band_gain_db(i, powerful3Preset[i])
				EQBands[i].value = EQ21Effect.get_band_gain_db(i)
		6:
			for i: int in EQ21Effect.get_band_count():
				EQ21Effect.set_band_gain_db(i, powerful4Preset[i])
				EQBands[i].value = EQ21Effect.get_band_gain_db(i)


func _on_music_player_finished() -> void:
	if loop:
		%MusicPlayer.play(0.0)
	else:
		next()


func get_track_part(url : String) -> String:
	var pos: int = url.find("track")
	if pos != -1:
		# Se "track" viene trovato, estrae tutto dal carattere successivo a "track"
		return "track" + url.substr(pos + len("track"))
	else:
		# Se "track" non viene trovato, restituisci l'URL originale
		return url


func _on_spotify_line_edit_text_submitted(new_text: String) -> void:
	downloadingTrackID = get_track_part(new_text).erase(0, 6)
	print(downloadingTrackID)

	%SpotifyLineEdit.editable = false
	%ImportSpotifyButton.disabled = true
	%downloadingLabel.text = "Downloading..."
	$MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Import/HBoxContainer/MarginContainer.show()

	%"TOKENsongTitle&AuthorRetriever".cancel_request()
	%"TOKENsongTitle&AuthorRetriever".request("https://open.spotify.com/get_access_token?reason=transport&productType=web_player")


func _on_toke_nsong_title_author_retriever_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray) -> void:
	if response_code == 200:
		var json: JSON = JSON.new()

		json.parse(body.get_string_from_utf8())

		print("Requested TOKEN is: ", json.get_data().accessToken)

		var finalAuthorName: String = authorNameToRequestImage.uri_encode()

		%"songTitle&AuthorRetriever".cancel_request()
		%"songTitle&AuthorRetriever".request("https://api.spotify.com/v1/tracks/".path_join(downloadingTrackID), ["Authorization: Bearer " + json.get_data().accessToken])
	else:
		printerr("Couldn't get TOKEN for getting downloading_author_image_and_title from Spotify API: ", response_code)


func _on_song_title_author_retriever_request_completed(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	var parsedBody: Dictionary = JSON.parse_string(body.get_string_from_ascii())
	if response_code == 200:
		print(parsedBody)

		var artist: String = parsedBody["album"]["artists"][0]["name"]
		var title: String = parsedBody["name"]

		downloadingSongAuthorName = artist
		downloadingSongName = title

		%downloadingProgress.start_checking()

		%spotifyDownloadHTTPRequest.request("https://yank.g3v.co.uk/track/".path_join(downloadingTrackID))
		print("Downloading song with URL: ", "https://yank.g3v.co.uk/track/".path_join(downloadingTrackID))
	else:
		printerr("Couldn't get AUTHOR & TITLE from Spotify API: ", response_code)
		print(parsedBody)


func _on_spotify_download_http_request_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray) -> void:
	%SpotifyLineEdit.editable = true
	%ImportSpotifyButton.disabled = false
	%downloadingProgress.stop_checking()
	%downloadingLabel.text = "Song Downloaded!"

	printt("Download finished!\n", result, response_code, headers, body.get_string_from_utf8())
	%SpotifyLineEdit.text = "Download finished!"

	check_songs_dir_exists()

	var file: FileAccess = FileAccess.open(OS.get_system_dir(OS.SYSTEM_DIR_MUSIC).path_join("GAMP-Downloaded").path_join(downloadingSongAuthorName + " - " + downloadingSongName + ".mp3"), FileAccess.WRITE)

	file.store_buffer(body)
	file.close()

	#dirSelectedImportSong(OS.get_system_dir(OS.SYSTEM_DIR_MUSIC).path_join("GAMP-Downloaded")

	importSingleSong(OS.get_system_dir(OS.SYSTEM_DIR_MUSIC).path_join("GAMP-Downloaded").path_join(downloadingSongAuthorName + " - " + downloadingSongName + ".mp3"))

	downloadingTrackID = ""
	downloadingSongAuthorName = ""
	downloadingSongName = ""


func _on_import_spotify_button_pressed() -> void:
	_on_spotify_line_edit_text_submitted(%SpotifyLineEdit.text)


func _on_manually_search_lyrics_button_pressed() -> void:
	if !manual_search_popup_control.visible:
		manual_search_popup_control.show()
	else:
		manual_search_popup_control.hide()


func _on_final_manual_search_button_pressed() -> void:
	if !manual_search_popup_control.visible:
		manual_search_popup_control.show()
	else:
		manual_search_popup_control.hide()

	# Search Lyrics with Name, Author, Duration inputs

	print("\n--- User asked to search for Lyrics: ---\n",
	"Song Name: " + %manualLyricsSearchSongNameLineEdit.text + "\n",
	"Song Author: " + %manualLyricsSearchSongNameAuthorLineEdit.text + "\n",
	"Song Duration: " + %manualLyricsSearchSongNameDurationLineEdit.text + "\n",
	)

	requestSongLyrics(%manualLyricsSearchSongNameLineEdit.text, %manualLyricsSearchSongNameAuthorLineEdit.text, %manualLyricsSearchSongNameDurationLineEdit.text)



func _on_song_cover_http_request_request_completed(_result: int, _response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:

	var json: JSON = JSON.new()

	json.parse(body.get_string_from_utf8())

	var imageURL : String = json.get_data().thumbnail_url

	print("Song Thumbnail Found: ", imageURL)

	%songCoverActualHTTPRequest.cancel_request()
	%songCoverActualHTTPRequest.request(imageURL)



func _on_song_cover_actual_http_request_request_completed(_result: int, _response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	var image : Image = Image.new()
	var error : Error = image.load_jpg_from_buffer(body)

	if error != OK:
		push_error("Couldn't load the image.")

	var texture : ImageTexture = ImageTexture.create_from_image(image)

	%songCoverTextureRect.texture = texture


func _on_check_for_text_file_timer_timeout() -> void:
	pass # Replace with function body.


func _on_lyrics_full_screen_button_pressed() -> void:
	if !lyricsFullscreen:
		lyricsFullscreen = true

		$"MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Song Info/HBoxContainer/HBoxContainer/VBoxContainer".hide()
		$"MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Song Info/HBoxContainer/HBoxContainer/VSeparator2".hide()
	else:
		lyricsFullscreen = false

		$"MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Song Info/HBoxContainer/HBoxContainer/VBoxContainer".show()
		$"MarginContainer/Panel/MarginContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/TabContainer/Song Info/HBoxContainer/HBoxContainer/VSeparator2".show()


func _on_stop_pressed() -> void:
	%MusicPlayer.play(0.0)
	%playButton.add_theme_font_size_override("font_size", 16)
	%playButton.text = "▶"
	%MusicPlayer.stream_paused = true
	pass # Replace with function body.


extends MarginContainer

@onready var songElementButton : Button = $SongElementButton
@onready var main : Control = $"../../../../../../../../../../.."
@onready var song_thumbnail_texture_rect: TextureRect = $Panel/MarginContainer/HBoxContainer/MarginContainer/PanelContainer/Panel/songThumbnailTextureRect
@onready var song_thumbnail_panel_container: PanelContainer = $Panel/MarginContainer/HBoxContainer/MarginContainer/PanelContainer/Panel/songThumbnailTextureRect/songThumbnailPanelContainer
@export var songFileName : String = "" # example: "author - song title.mp3"
@export var songFileNamePath : String = "" # example: "C:\user\desktop\musicfolder\author - song title.mp3"
@export var songFileNameDir : String = " " # example: "C:\user\desktop\musicfolder\"

var currentSongTimestamp: float = 0.0

var playing : bool = false

var hover : bool = false

var songMetadataExtractor: MusicMetadata = MusicMetadata.new()

var songFileData: PackedByteArray

var setCardInfosThread: Thread = Thread.new()

signal songElementSelected(songElementNode : Node, songFileName : String, songFileNamePath : String, songFileNameDir : String, songAuthor : String, songTitle : String, songTotalDuration : String, songCurrentTimestamp: float)

signal infoImportCompleted()

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	songElementSelected.connect(main.songElementSelectedFunction)
	#connect("songElementSelected", Callable(main, "songElementSelectedFunction"))
	pivot_offset = size / 2.0
	$Panel.pivot_offset = $Panel.size / 2.0
	songElementButton.pivot_offset = songElementButton.size / 2.0
	
	setCardInfosThread.start(setCardInfosThreaded, Thread.PRIORITY_HIGH)

func setCardInfosThreaded() -> void:
	songFileData = FileAccess.get_file_as_bytes(songFileNamePath)
	
	# Estraiamo i metadati
	var title: String = get_title(songFileData)
	var artist: String = get_artist(songFileData)
	var album_artist: String = get_album_artist(songFileData)
	var album: String = get_album(songFileData)
	
	# Estraiamo e processiamo la cover
	var cover_data := get_cover(songFileData)
	var image: Image
	
	if !cover_data.is_empty():
		image = create_image_from_cover_data(cover_data)
		if image:
			image.resize(400, 400, Image.INTERPOLATE_LANCZOS)
			image.compress(Image.COMPRESS_BPTC, Image.COMPRESS_SOURCE_GENERIC)
			var texture: Texture2D = ImageTexture.create_from_image(image)
			song_thumbnail_texture_rect.set_deferred("texture", texture)
	
	# Impostiamo i metadati nell'interfaccia
	call_deferred("setTitle", title if title else songFileName)
	call_deferred("setAuthor", artist if artist else album_artist)
	
	call_deferred("thread_completed_info_import")

func thread_completed_info_import() -> void:
	setCardInfosThread.wait_to_finish()
	infoImportCompleted.emit()

#region Metadata Parsing Region
# Funzione per interpretare i byte unsynchsafe
func unsynchsafe(buffer: PackedByteArray) -> int:
	return (buffer[0] << 21) | (buffer[1] << 14) | (buffer[2] << 7) | buffer[3]

func read_id3v2_header(data: PackedByteArray) -> Dictionary:
	var header := {
		"version": 0,
		"revision": 0,
		"flags": 0,
		"size": 0,
		"valid": false
	}
	
	if data.size() < 10:
		print("File troppo piccolo per contenere un header ID3")
		return header
		
	var id3_header: PackedByteArray = data.slice(0, 10)
	if id3_header.slice(0, 3).get_string_from_ascii() != "ID3":
		print("Non è un file ID3")
		return header
		
	header.version = id3_header[3]
	header.revision = id3_header[4]
	header.flags = id3_header[5]
	header.size = unsynchsafe(id3_header.slice(6, 10))
	header.valid = true
	
	print("ID3v2.", header.version, ".", header.revision, " flags:", header.flags, " size:", header.size)
	return header

func read_frame(data: PackedByteArray, frame_id: String) -> PackedByteArray:
	var pos: int = 10
	var header := read_id3v2_header(data)
	var unsync: bool = header.flags & 0x80 > 0
	
	while pos < data.size():
		var frame_header: PackedByteArray = data.slice(pos, pos + 10)
		var id: String = frame_header.slice(0, 4).get_string_from_ascii()
		# Gestione speciale per APIC e controllo sync-safe
		var Size: int = unsynchsafe(frame_header.slice(4, 8))
		if Size > 0x7f and id == "APIC":
			Size = (frame_header[4] << 24) | (frame_header[5] << 16) | (frame_header[6] << 8) | frame_header[7]
		
		pos += 10
		if id == frame_id:
			var frame_data := data.slice(pos, pos + Size)
			# Applica desincronizzazione se necessario
			if unsync:
				frame_data = unsynchronize(frame_data)
			return frame_data
		pos += Size
	return PackedByteArray()

func unsynchronize(data: PackedByteArray) -> PackedByteArray:
	var result := PackedByteArray()
	var i := 0
	while i < data.size():
		result.append(data[i])
		if data[i] == 0xFF and i + 1 < data.size() and data[i + 1] == 0x00:
			i += 1
		i += 1
	return result

func get_rating(data: PackedByteArray) -> int:
	# Prova a cercare il frame POPM per la classificazione
	var rating_frame: PackedByteArray = read_frame(data, "POPM")
	
	print("Contenuto frame POPM:", rating_frame)
	
	if rating_frame.size() > 0:
		print("Contenuto frame POPM:", rating_frame)
		return int(rating_frame[0])  # Primo byte del frame POPM
	
	# Se il frame POPM non è presente, prova a cercare nei frame TXXX
	print("Frame POPM non presente, si procede a cercare nei frame TXXX")
	var pos: int = 10  # Inizia subito dopo l'header ID3v2
	while pos < data.size():
		var frame_header: PackedByteArray = data.slice(pos, pos + 10)
		var id: String = frame_header.slice(0, 4).get_string_from_ascii()
		var Size: int = unsynchsafe(frame_header.slice(4, 8))
		pos += 10
		
		if id == "TXXX":
			# Legge il contenuto del frame TXXX
			var txxx_content: PackedByteArray = data.slice(pos, pos + Size)
			var description: String = txxx_content.get_string_from_ascii()
			
			# Se la descrizione contiene "RATING" o "CLASSIFICATION"
			if description.find("RATING") != -1 or description.find("CLASSIFICATION") != -1:
				# Ritorna il primo byte dopo la descrizione come rating
				print("Frame TXXX con RATING trovato:", txxx_content)
				return int(txxx_content[description.length() + 1])  # Valore dopo la descrizione
		
		pos += Size  # Passa al prossimo frame

	# Nessuna classificazione trovata
	print("Classificazione non trovata né in POPM né nei TXXX")
	return -1


func get_duration(data: PackedByteArray) -> int:
	var duration_frame: PackedByteArray = read_frame(data, "TLEN")
	if duration_frame.size() > 0:
		print("Contenuto frame TLEN:", duration_frame)
		return int(duration_frame.get_string_from_ascii())
	print("Frame TLEN non presente")
	return -1

func get_cover(data: PackedByteArray) -> Dictionary:
	var header := read_id3v2_header(data)
	if !header.valid:
		return {}
		
	var covers := []
	var pos := 10  # Dopo l'header ID3
	
	while pos < header.size + 10:  # +10 per includere la dimensione dell'header
		if pos + 10 > data.size():
			break
			
		var frame_header := data.slice(pos, pos + 10)
		var frame_id := frame_header.slice(0, 4).get_string_from_ascii()
		var frame_size := unsynchsafe(frame_header.slice(4, 8))
		var frame_flags := (frame_header[8] << 8) | frame_header[9]
		
		pos += 10
		
		if frame_id == "APIC":
			if pos + frame_size > data.size():
				print("Frame APIC troncato")
				break
				
			var frame_data := data.slice(pos, pos + frame_size)
			var cover := parse_apic_frame(frame_data, header.version)
			if !cover.is_empty():
				covers.append(cover)
		
		pos += frame_size
	
	# Scegliamo la cover migliore disponibile
	return select_best_cover(covers)

func parse_apic_frame(frame_data: PackedByteArray, id3_version: int) -> Dictionary:
	var result := {
		"mime_type": "",
		"picture_type": 0,
		"description": "",
		"image_data": PackedByteArray()
	}
	
	if frame_data.is_empty():
		return {}
	
	var encoding := frame_data[0]
	var pos := 1
	
	# Cerca la fine del MIME type
	var mime_end := frame_data.find(0, pos)
	if mime_end != -1:
		result.mime_type = frame_data.slice(pos, mime_end).get_string_from_ascii()
		pos = mime_end + 1
		
		if pos < frame_data.size():
			result.picture_type = frame_data[pos]
			pos += 1
			
			# Salta la descrizione in base all'encoding
			match encoding:
				0:  # ASCII
					var desc_end := frame_data.find(0, pos)
					pos = desc_end + 1 if desc_end != -1 else pos
				1, 2:  # UTF-16
					while pos < frame_data.size() - 1:
						if frame_data[pos] == 0 and frame_data[pos + 1] == 0:
							pos += 2
							break
						pos += 1
				3:  # UTF-8
					var desc_end := frame_data.find(0, pos)
					pos = desc_end + 1 if desc_end != -1 else pos
	
	# Cerca l'inizio effettivo dei dati dell'immagine
	var image_start := pos
	while image_start < frame_data.size() - 2:
		# Cerca la signature JPEG
		if frame_data[image_start] == 0xFF and frame_data[image_start + 1] == 0xD8:
			result.mime_type = "image/jpeg"
			result.image_data = frame_data.slice(image_start)
			print("JPEG trovato alla posizione:", image_start)
			print("Dimensione dati immagine:", result.image_data.size())
			# Verifica che l'immagine termini correttamente con EOI marker
			var has_eoi := false
			for i in range(result.image_data.size() - 2):
				if result.image_data[i] == 0xFF and result.image_data[i + 1] == 0xD9:
					has_eoi = true
					# Taglia via eventuali dati extra dopo l'EOI
					result.image_data = result.image_data.slice(0, i + 2)
					break
			if !has_eoi:
				print("Warning: JPEG senza marker EOI valido")
			break
			
		# Cerca la signature PNG
		elif frame_data[image_start] == 0x89 and frame_data[image_start + 1] == 0x50:
			result.mime_type = "image/png"
			result.image_data = frame_data.slice(image_start)
			print("PNG trovato alla posizione:", image_start)
			break
			
		image_start += 1
	
	return result

func create_image_from_cover_data(cover_data: Dictionary) -> Image:
	if cover_data.is_empty() or cover_data.image_data.is_empty():
		print("Dati cover vuoti o mancanti")
		return null
		
	var image := Image.new()
	var error: Error
	
	print("Tentativo di caricamento immagine di tipo:", cover_data.mime_type)
	print("Dimensione dati:", cover_data.image_data.size())
	print("Primi byte:", cover_data.image_data.slice(0, 4))
	
	# Verifica che i dati inizino con i marker corretti
	if cover_data.mime_type.begins_with("image/jp"):  # jpeg o jpg
		if cover_data.image_data.size() < 2 or cover_data.image_data[0] != 0xFF or cover_data.image_data[1] != 0xD8:
			print("Errore: Dati JPEG non validi - Header mancante")
			return null
			
		error = image.load_jpg_from_buffer(cover_data.image_data)
		if error != OK:
			print("Errore caricamento JPEG:", error)
			# Prova a salvare i dati per debug
			var file := FileAccess.open("user://debug_cover.jpg", FileAccess.WRITE)
			if file:
				file.store_buffer(cover_data.image_data)
				print("Dati JPEG salvati in user://debug_cover.jpg per debug")
			return null
			
	elif cover_data.mime_type == "image/png":
		if cover_data.image_data.size() < 8 or cover_data.image_data[0] != 0x89 or cover_data.image_data[1] != 0x50:
			print("Errore: Dati PNG non validi - Header mancante")
			return null
			
		error = image.load_png_from_buffer(cover_data.image_data)
		if error != OK:
			print("Errore caricamento PNG:", error)
			return null
	else:
		print("Tipo MIME non supportato:", cover_data.mime_type)
		return null
		
	if error == OK:
		print("Immagine caricata con successo")
		return image
	else:
		print("Errore generico nel caricamento dell'immagine:", error)
		return null

func select_best_cover(covers: Array) -> Dictionary:
	if covers.is_empty():
		return {}
	
	# Priorità dei tipi di immagine (secondo lo standard ID3v2)
	var type_priority := {
		3: 0,   # Cover frontale
		2: 1,   # File icon
		1: 2,   # Icon 32x32
		0: 3,   # Other
		4: 4,   # Cover posteriore
	}
	
	# Priorità dei formati immagine
	var mime_priority := {
		"image/jpeg": 0,
		"image/jpg": 0,
		"image/png": 1,
		"image/gif": 2
	}
	
	var best_cover: Dictionary = covers[0]
	var best_priority := 999
	
	for cover in covers:
		var type_prio: int = type_priority.get(cover.picture_type, 999)
		var mime_prio: int = mime_priority.get(cover.mime_type.to_lower(), 999)
		var current_priority: int = type_prio * 1000 + mime_prio
		
		if current_priority < best_priority:
			best_priority = current_priority
			best_cover = cover
		elif current_priority == best_priority and cover.image_data.size() > best_cover.image_data.size():
			# A parità di priorità, scegli l'immagine più grande
			best_cover = cover
	
	return best_cover

# Funzione generica per leggere i frame di testo
func read_text_frame(data: PackedByteArray, frame_id: String) -> String:
	var text_frame: PackedByteArray = read_frame(data, frame_id)
	if text_frame.size() > 0:
		# Il primo byte è l'encoding
		var encoding: int = text_frame[0]
		var text_data: PackedByteArray = text_frame.slice(1)
		
		match encoding:
			0:  # ISO-8859-1
				return text_data.get_string_from_ascii()
			1:  # UTF-16 con BOM
				# Rimuoviamo il BOM e decodifichiamo come UTF-16
				if text_data.size() >= 2:
					return text_data.slice(2).get_string_from_utf16()
			2:  # UTF-16BE
				return text_data.get_string_from_utf16()
			3:  # UTF-8
				return text_data.get_string_from_utf8()
		
		# Fallback a ASCII se l'encoding non è gestito
		return text_data.get_string_from_ascii()
	
	return ""

# Funzione per ottenere il titolo
func get_title(data: PackedByteArray) -> String:
	var title := read_text_frame(data, "TIT2")
	if title.is_empty():
		print("Nessun titolo trovato")
		return ""
	
	print("Titolo trovato:", title)
	return title.strip_edges()

# Funzione per ottenere l'artista
func get_artist(data: PackedByteArray) -> String:
	var artist := read_text_frame(data, "TPE1")
	if artist.is_empty():
		print("Nessun artista trovato")
		return ""
	
	print("Artista trovato:", artist)
	return artist.strip_edges()

# Funzione per ottenere l'album
func get_album(data: PackedByteArray) -> String:
	var album := read_text_frame(data, "TALB")
	if album.is_empty():
		print("Nessun album trovato")
		return ""
	
	print("Album trovato:", album)
	return album.strip_edges()

# Funzione per ottenere l'artista dell'album
func get_album_artist(data: PackedByteArray) -> String:
	var album_artist := read_text_frame(data, "TPE2")
	if album_artist.is_empty():
		print("Nessun artista dell'album trovato")
		return ""
	
	print("Artista dell'album trovato:", album_artist)
	return album_artist.strip_edges()

#endregion

func songElementPressed() -> void:
	pass

func _on_song_element_button_mouse_entered() -> void:
	hover = true
	
	var tween: Tween = create_tween()
	
	tween.set_ease(Tween.EASE_IN_OUT)
	tween.set_process_mode(Tween.TWEEN_PROCESS_PHYSICS)
	tween.set_trans(Tween.TRANS_QUAD)
	
	tween.tween_property(self, "scale", Vector2(0.98, 0.98), .15).from_current()
	tween.parallel().tween_property(self, "modulate", Color.WHITE * 1.175, .15).from_current()
	#tween.chain().tween_property(songElementButton, "scale", Vector2(0.98, 0.98), .15)
	
	$Panel/Panel.show()


func _on_song_element_button_mouse_exited() -> void:
	hover = false
	
	var tween: Tween = create_tween()
	
	tween.set_ease(Tween.EASE_IN_OUT)
	tween.set_process_mode(Tween.TWEEN_PROCESS_PHYSICS)
	tween.set_trans(Tween.TRANS_QUAD)
	
	tween.tween_property(self, "scale", Vector2(1.0, 1.0), .15).from_current()
	
	if playing:
		tween.parallel().tween_property(self, "modulate", Color.WHITE * 1.15, .2).from_current()
	else:
		tween.parallel().tween_property(self, "modulate", Color.WHITE, .15).from_current()
	
	#tween.chain().tween_property(songElementButton, "scale", Vector2(1.0, 1.0), .15)
	
	$Panel/Panel.hide()


func _on_song_element_button_pressed() -> void:
	if !main.smooth_scroll_container.velocity > Vector2.ZERO:
		songElementButton.grab_focus()
		songElementSelected.emit(self, songFileName, songFileNamePath, songFileNameDir, %Author.text, %SongTitle.text, %TotalDuration.text, currentSongTimestamp)
		
		var tween: Tween = create_tween()
		
		tween.set_ease(Tween.EASE_IN_OUT)
		tween.set_process_mode(Tween.TWEEN_PROCESS_PHYSICS)
		tween.set_trans(Tween.TRANS_QUAD)
		
		tween.parallel().tween_property(self, "modulate", Color.WHITE * 1.15, 0.2).from_current()


func _on_song_element_button_button_up() -> void:
	songElementPressed()
	var tween: Tween = create_tween()
	
	tween.set_ease(Tween.EASE_IN_OUT)
	tween.set_process_mode(Tween.TWEEN_PROCESS_IDLE)
	tween.set_trans(Tween.TRANS_QUAD)
	
	tween.tween_property(self, "scale", Vector2(1.0, 1.0), .15)
	tween.chain().tween_property(songElementButton, "scale", Vector2(1.0, 1.0), .15)


func _on_song_element_button_button_down() -> void:
	var tween: Tween = create_tween()
	
	tween.set_ease(Tween.EASE_IN_OUT)
	tween.set_process_mode(Tween.TWEEN_PROCESS_IDLE)
	tween.set_trans(Tween.TRANS_QUAD)
	
	tween.tween_property(self, "scale", Vector2(0.95, 0.95), .15)
	tween.chain().tween_property(songElementButton, "scale", Vector2(0.95, 0.95), .15)

func setTitle(title : String) -> void:
	%SongTitle.text = title

func setAuthor(Author : String) -> void:
	%Author.text = Author

func setProgressBarValue(value : float) -> void:
	%songProgressBar.value = value

func setCurrentDuration(currentDuration : String) -> void:
	%CurrentDuration.text = currentDuration

func setTotalDuration(totalDuration : String) -> void:
	%TotalDuration.text = totalDuration

func setSongFileName(songName : String) -> void:
	songFileName = songName

func setSongFileNamePath(songPath : String) -> void:
	songFileNamePath = songPath

func setSongFileNameDir(songDir : String) -> void:
	songFileNameDir = songDir

func songElementSelectedThumbnailOpacityOnAnimation() -> void:
	var tween : Tween = create_tween()
	
	tween.set_ease(Tween.EASE_IN_OUT)
	tween.set_process_mode(Tween.TWEEN_PROCESS_IDLE)
	tween.set_trans(Tween.TRANS_QUAD)
	
	tween.tween_property(song_thumbnail_panel_container, "self_modulate", Color(0, 0, 0, 0.5), 0.35)

func songElementSelectedThumbnailOpacityOffAnimation() -> void:
	var tween : Tween = create_tween()
	
	tween.set_ease(Tween.EASE_IN_OUT)
	tween.set_process_mode(Tween.TWEEN_PROCESS_IDLE)
	tween.set_trans(Tween.TRANS_QUAD)
	
	tween.tween_property(song_thumbnail_panel_container, "self_modulate", Color(0, 0, 0, 0), 0.35)

func playPlayingAnimation() -> void:
	#$Panel/Label.show()
	%linesAnimationPlayer.play("playing")
	songElementSelectedThumbnailOpacityOnAnimation()
	$Panel/MarginContainer/HBoxContainer/MarginContainer/PanelContainer/Panel/HBoxContainer.show()

func stopPlayingAnimation() -> void:
	#$Panel/Label.hide()
	%linesAnimationPlayer.play("RESET")
	songElementSelectedThumbnailOpacityOffAnimation()
	$Panel/MarginContainer/HBoxContainer/MarginContainer/PanelContainer/Panel/HBoxContainer.hide()


func _on_song_element_button_focus_entered() -> void:
	$Panel/Panel.show()

func _on_song_element_button_focus_exited() -> void:
	$Panel/Panel.hide()


func _on_resized() -> void:
	pivot_offset = size / 2.0


extends ColorRect

#Save this for a class Animation

@export var VU_COUNT: int = 30
@export var FREQ_MAX: float = 11050.0
@export var MIN_DB: float = 60
@export var ANIMATION_SPEED: float = 0.1
@export var HEIGHT_SCALE: float = 8.0

var spectrum: AudioEffectSpectrumAnalyzerInstance
var min_values: PackedFloat32Array = []
var max_values: PackedFloat32Array = []
var timer = 50
var timer_count = 0


func _ready() -> void:
	spectrum = AudioServer.get_bus_effect_instance(0, 0)
	min_values.resize(VU_COUNT)
	min_values.fill(0.0)
	max_values.resize(VU_COUNT)
	max_values.fill(0.0)


func _process(_delta: float) -> void:

	'''  20 HZ  timer = 50 micro sec '''



	timer_count = Time.get_ticks_msec()
	if timer_count >= timer:
		timer += 50

		var prev_hz: float = 0
		var data: PackedFloat32Array = []
		for i: int in range(1, VU_COUNT + 1):
			var hz: float = i * FREQ_MAX / VU_COUNT
			var f: Vector2 = spectrum.get_magnitude_for_frequency_range(prev_hz, hz)
			var energy: float = clamp((MIN_DB + linear_to_db(f.length())) / MIN_DB, 0.0, 1.0)
			data.append(energy * HEIGHT_SCALE)
			prev_hz = hz
		for i: int in range(VU_COUNT):
			if data[i] > max_values[i]:
				max_values[i] = data[i]
			else:
				max_values[i] = lerp(max_values[i], data[i], ANIMATION_SPEED)
			if data[i] <= 0.0:
				min_values[i] = lerp(min_values[i], 0.0, ANIMATION_SPEED)
		var fft: PackedFloat32Array = []
		for i: int in range(VU_COUNT):
			fft.append(lerp(min_values[i], max_values[i], ANIMATION_SPEED))
		get_material().set_shader_parameter("freq_data", fft)
		pass


func _on_loal_pressed() -> void:
	prints("stac")
	pass # Replace with function body.



Detectar idioma

inglés

español

Francés

español

inglés

Francés
The resulting model is significantly more efficient than the original GRU, requiring only O(2dhdx)
parameters, compared to GRU’s O(3dh(dx + dh)) parameters, where dx and dh denote the sizes of
the input xt and the hidden state ht, respectively. In RNNs, state expansion is often used (i.e., dh =
αdx, where α ≥ 1), which helps the models better capture features from the input data. minGRU
uses approximately 33%, 22%, 17%, and 13% of the parameters of a GRU when α = 1,2,3,4,
respectively.
Additionally, the minimal version of GRU can now be trained in parallel using the parallel scan
algorithm, bypassing the need for backpropagation through time (BPTT). Pseudocode and a simple
PyTorch implementation are included in the Appendix.
3.2 AMinimalLSTM:minLSTM
3.2.1 Step 1: Drop previous state dependencies from gates
Revisiting LSTM’s cell state recurrence which works as follows:
ct = ft ⊙ct−1 +it ⊙ ˜ct
Similar to GRU’s hidden state, we can see that LSTM’s cell state recurrence resembles the afore
mentioned parallel scan’s formulation vt = at ⊙ vt−1 + bt where at ← ft, bt ← it ⊙ ˜ct, and
vt ← ct. However, ft, it and ˜ct are dependent on the previous hidden state ht. As such, LSTM’s
cell state recurrence is unable to apply the parallel scan algorithm as is. We can address this in a
similar fashion to GRU by removing their hidden state dependencies as follows:
ft = σ(Lineardh
([xt,ht−1]))
it = σ(Lineardh
([xt,ht−1]))
˜
ct = tanh(Lineardh
([xt,ht−1]))
⇒
ft = σ(Lineardh
(xt))
it = σ(Lineardh
(xt))
˜
ct = tanh(Lineardh
(xt))
3.2.2 Step 2: Drop range restriction of candidate states
Similar to GRUs, LSTMs leverage the hyperbolic tangent function (tanh) to restrict the range of
its states between (−1,1). LSTMs apply the range restriction twice: once when computing the
candidate cell state and once when computing its hidden state. In this step, we drop both as follows:
˜
ct = tanh(Lineardh
(xt))
ht = ot ⊙tanh(ct)
3.2.3 Step 3: Simplifying scaling of output
⇒ ˜ct =Lineardh
(xt)
ht = ot ⊙ct
Continuing the trend of simplification, we drop the output gate ot which scales the hidden state.
Without the output gate, the normalized hidden state is equal to the cell state, i.e., ht = ot ⊙ ct ⇒
ht = ct, making having both a hidden and cell state unnecessary. As such, we drop the cell state as
well, resulting in the following modification:
5
ht = ot ⊙ct
ot = σ(Lineardh
(xt))
ct = ft ⊙ct−1 +it ⊙ ˜ct
˜
ct = Lineardh
(xt)
⇒ ht=ft⊙ht−1+it⊙˜ ht
˜
ht = Lineardh
(xt)
In manysequencemodelling settings (e.g., text generation), the optimization objective/target is time
independent in scale. Recall LSTM’s cell state recurrence ct = ft ⊙ ct−1 + it ⊙ ˜ct where it,ft ∈
(0, 1)dh, and GRU’s hidden state recurrence1, hGRU
t
= (1−zt) ⊙hGRU
t−1 +zt ⊙ ˜ hGRU
where
zt ∈ (0,1)dh. GRUs retain (1 − zt) ∈ (0,1) of the previous hidden state and add zt of the
new candidate state. Since these proportions sum to 1, the model ensures its outputs (i.e., hidden
states) are time-independent in scale. In contrast, LSTM’s forget and input gates are computed
independently (e.g., ft,it → 1 or ft,it → 0), making its states time-dependent in scale2. For
tasks where time-independence is important, we can ensure LSTM’s output is time-independent in
scale by simply normalizing its input and forget gates, i.e., f′
t,i′
t ← ft
ft+it 
, it
t
ft+it 
, ensuring that
f′
t + i′
t = 1 and the scale of LSTM’s state is time-independent.
3.2.4 minLSTM
Combining the three steps results in a minimal version of LSTM (minLSTM):
LSTM
ht = ot ⊙tanh(ct)
ot = σ(Lineardh
([xt,ht−1]))
ct = ft ⊙ct−1 +it ⊙ ˜ct
ft = σ(Lineardh
([xt,ht−1]))
it = σ(Lineardh
([xt,ht−1]))
˜
ct = tanh(Lineardh
([xt,ht−1]))
⇒
minLSTM
ht = ft ⊙ht−1 +it ⊙ ˜ ht
ft = σ(Lineardh
(xt))
it = σ(Lineardh
(xt))
˜
ht = Lineardh
(xt)
where time-independent outputs can be achieved using a hidden state recurrence ht = f′
t ⊙ht−1 +
i′
t ⊙ ˜ ht with normalized forget f′
t and input it gates computed as f′
t,i′
t ← ft
ft+it 
, it
ft+it 
.
The resulting model is significantly more efficient than the original LSTM, requiring only O(3dhdx)
parameters compared to LSTM’s O(4dh(dx + dh)). Considering state expansion (i.e., dh = αdx,
where α ≥ 1),minLSTMusesapproximately 38%,25%,19%,or15%oftheparameters of a LSTM
when α = 1,2,3, or 4 respectively.
Additionally, the minimal version of LSTM can now be trained in parallel using the parallel scan
algorithm, bypassing the need for backpropagation through time (BPTT). Pseudocode and a simple
PyTorch implementation are included in the Appendix.
4 WereRNNsAllWeNeeded?
In this section, we compare the minimal versions (minLSTMs and minGRUs) with their traditional
counterparts (LSTMs and GRUs) and modern sequence models. Pseudocode, PyTorch implementa
tion, and detailed information regarding the experiment setup are available in the Appendix
4.829
El modelo resultante es significativamente más eficiente que el GRU original, ya que requiere solo O(2dhdx) parámetros, en comparación con los O(3dh(dx + dh)) parámetros del GRU, donde dx y dh denotan los tamaños de la entrada xt y el estado oculto ht, respectivamente. En las RNN, se utiliza a menudo la expansión de estado (es decir, dh = αdx, donde α ≥ 1), lo que ayuda a los modelos a capturar mejor las características de los datos de entrada. minGRU utiliza aproximadamente el 33 %, 22 %, 17 % y 13 % de los parámetros de un GRU cuando α = 1, 2, 3, 4, respectivamente.
Además, la versión mínima de GRU ahora puede entrenarse en paralelo mediante el algoritmo de escaneo paralelo, lo que evita la necesidad de retropropagación en el tiempo (BPTT). El pseudocódigo y una implementación sencilla de PyTorch se incluyen en el Apéndice. 3.2 AMinimalLSTM:minLSTM
3.2.1 Paso 1: Eliminar las dependencias de estados previos de las puertas
Revisando la recurrencia de estados de celda de LSTM, que funciona de la siguiente manera:
ct = ft ⊙ct−1 +it ⊙ ˜ct
Similar al estado oculto de GRU, podemos ver que la recurrencia de estados de celda de LSTM se asemeja a la formulación de escaneo paralelo mencionada anteriormente: vt = at ⊙ vt−1 + bt, donde at ← ft, bt ← it ⊙ ˜ct, y vt ← ct. Sin embargo, ft, it y ˜ct dependen del estado oculto previo ht. Por lo tanto, la recurrencia de estados de celda de LSTM no puede aplicar el algoritmo de escaneo paralelo tal como está. Podemos abordar esto de forma similar a GRU, eliminando sus dependencias de estado ocultas de la siguiente manera:
ft = σ(Lineardh
([xt,ht−1]))
it = σ(Lineardh
([xt,ht−1]))
˜
ct = tanh(Lineardh
([xt,ht−1]))
⇒
ft = σ(Lineardh
(xt))
it = σ(Lineardh
(xt))
˜
ct = tanh(Lineardh
(xt))
3.2.2 Paso 2: Eliminar la restricción de rango de los estados candidatos
De forma similar a las GRU, las LSTM utilizan la función tangente hiperbólica (tanh) para restringir el rango de sus estados entre (−1,1). Las LSTM aplican la restricción de rango dos veces: una al calcular el estado de la celda candidata y otra al calcular su estado oculto. En este paso, eliminamos ambos valores de la siguiente manera:
˜
ct = tanh(Lineardh
(xt))
ht = ot ⊙tanh(ct)
3.2.3 Paso 3: Simplificación del escalado de la salida
⇒ ˜ct = Lineardh
(xt)
ht = ot ⊙ct
Continuando con la simplificación, eliminamos la puerta de salida ot, que escala el estado oculto.
Sin la puerta de salida, el estado oculto normalizado es igual al estado de la celda, es decir, ht = ot ⊙ ct ⇒
ht = ct, lo que hace innecesario tener un estado oculto y uno de celda. Por lo tanto, también eliminamos el estado de la celda, lo que resulta en la siguiente modificación:
5
ht = ot ⊙ct
ot = σ(Lineardh
(xt))
ct = ft ⊙ct−1 +it ⊙ ˜ct
˜
ct = Lineardh
(xt)
⇒ ht=ft⊙ht−1+it⊙˜ ht
˜
ht = Lineardh
(xt)
En muchos entornos de modelado de secuencias (por ejemplo, generación de texto), el objetivo de optimización es independiente del tiempo en escala. Recordemos la recurrencia del estado de celda de LSTM ct = ft ⊙ ct−1 + it ⊙ ˜ct donde it,ft ∈ (0, 1)dh, y la recurrencia del estado oculto de GRU 1, hGRU
t
= (1−zt) ⊙hGRU
t−1 +zt ⊙ ˜ hGRU
donde zt ∈ (0, 1)dh. Los GRU conservan (1 − zt) ∈ (0, 1) del estado oculto previo y añaden zt del nuevo estado candidato. Dado que estas proporciones suman 1, el modelo garantiza que sus salidas (es decir, estados ocultos) sean independientes del tiempo en escala. Por el contrario, las puertas de entrada y olvido de LSTM se calculan de forma independiente (p. ej., ft,it → 1 o ft,it → 0), lo que hace que sus estados dependan del tiempo en la escala 2. Para tareas donde la independencia temporal es importante, podemos garantizar que la salida de LSTM sea independiente del tiempo en la escala simplemente normalizando sus puertas de entrada y olvido, es decir, f′
t,i′
t ← ft
ft+it
, it
t
ft+it
, asegurando que f′
t + i′
t = 1 y que la escala del estado de LSTM sea independiente del tiempo. 3.2.4 minLSTM
La combinación de los tres pasos da como resultado una versión mínima de LSTM (minLSTM):
LSTM
ht = ot ⊙tanh(ct)
ot = σ(Lineardh
([xt,ht−1]))
ct = ft ⊙ct−1 +it ⊙ ˜ct
ft = σ(Lineardh
([xt,ht−1]))
it = σ(Lineardh
([xt,ht−1]))
˜
ct = tanh(Lineardh
([xt,ht−1]))
⇒
minLSTM
ht = ft ⊙ht−1 +it ⊙ ˜ ht
ft = σ(Lineardh
(xt))
it = σ(Lineardh
(xt))
˜
ht = Linear h
(xt)
donde se pueden obtener salidas independientes del tiempo utilizando una recurrencia de estado oculta ht = f′
t ⊙ ht−1 +
i′
t ⊙ ˜ ht con compuertas de olvido de f′
t normalizadas y de entrada it calculadas como f′
t, i′
t ← ft
ft+it
, it
ft+it
.
El modelo resultante es significativamente más eficiente que el LSTM original, requiriendo solo O(3dhdx)
parámetros en comparación con los O(4dh(dx + dh) del LSTM). Considerando la expansión de estado (es decir, dh = αdx,
donde α ≥ 1), minLSTM utiliza aproximadamente el 38 %, 25 %, 19 % o 15 % de los parámetros de un LSTM
cuando α = 1, 2, 3 o 4 respectivamente. Además, la versión mínima de LSTM ahora puede entrenarse en paralelo mediante el algoritmo de escaneo paralelo, lo que evita la necesidad de retropropagación en el tiempo (BPTT). El pseudocódigo y una implementación sencilla de PyTorch se incluyen en el Apéndice.

4 ¿Eran las RNN todo lo que necesitábamos?

En esta sección, comparamos las versiones mínimas (minLSTM y minGRU) con sus contrapartes tradicionales (LSTM y GRU) y con los modelos de secuencia modernos. El pseudocódigo, la implementación de PyTorch y la información detallada sobre la configuración del experimento están disponibles en el Apéndice.
Enviar comentarios
Pulsa el tabulador para acceder a las acciones
/**/*/*/*


Detectar idioma

inglés

español

Francés

español

inglés

Francés
3 Methodology
Interestingly, we can see that the GRU’s hidden state and LSTM’s cell state recurrences resemble the
vector formulation. In this section, we demonstrate that GRUs and LSTMs are trainable via parallel
scan by removing their previous state dependencies from their various gates. Building on this, we
further simplify these RNNs by removing their constraints on output range (i.e., tanh). Combining
the steps, we describe minimal versions of GRUs and LSTMs (minGRUs and minLSTMs) that are
trainable in parallel.
3
3.1 AMinimalGRU:minGRU
3.1.1 Step 1: Drop previous state dependencies from gates
Revisiting GRU’s hidden state recurrence which works as follows:
ht = (1−zt)⊙ht−1 +zt ⊙ ˜ ht
We can observe that the recurrence resembles the aforementioned parallel scan’s formulation vt =
at ⊙ vt−1 + bt where at ← (1 − zt), bt ← zt ⊙ ˜ ht, and vt ← ht. However, zt and ˜ ht
are dependent on the previous hidden state ht−1, i.e., zt = σ(Lineardh
([xt,ht−1])) and ˜ ht =
tanh(Lineardh
([xt,rt ⊙ ht−1])). As a result, it is not possible to apply the parallel scan as is since
the algorithm’s inputs a1,...,an and b1,...,bn are conditional on already knowing its outputs
h1,...,hn−1.
A simple remedy to this is to simplify GRU by removing their previous hidden state (i.e., ht−1)
dependencies. Specifically, the changes are as follows:
zt = σ(Lineardh
([xt,ht−1]))
rt = σ(Lineardh
([xt,ht−1]))
˜
ht = tanh(Lineardh
([xt,rt ⊙ ht−1]))
⇒ zt=σ(Lineardh
(xt))
˜
ht = tanh(Lineardh
(xt))
By removing the dependence on ht−1 from the candidate hidden state ˜ ht, the reset gate rt that
would control ht−1 weight is also no longer needed and is removed. Without the dependencies on
previous hidden states, the inputs to the algorithm a1,...,an and b1,...,bn are all easily computed
in parallel and can thus be used to compute h1,...,hn efficiently via the parallel scan.
Although there have been theoretical concerns about the absence of previous state dependen
cies (Merrill et al., 2024), there has also been substantial empirical evidence supporting the effec
tiveness of models that omit these dependencies, such as xLSTM (Beck et al., 2024) and Mamba (Gu
&Dao,2024). Instead of explicitly modelling dependencies on previous states to capture long-range
dependencies, these kinds of recurrent models can learn them by stacking multiple layers. Notably,
in the xLSTM paper, their fully parallelized version (xLSTM[1:0]), which eliminates hidden state
dependencies, performed similarly to — and in some cases, better than — versions that retain these
dependencies (e.g., xLSTM[7:1]).
3.1.2 Step 2: Drop range restriction of candidate states
In GRU’shidden state recurrence, the proportion carried over from the previous hidden state (1−zt)
and the amount added for the new candidate hidden state (zt) sum to 1. As a result, the scale of
GRU’s hidden state value is time-independent. Instead, the scale of its hidden state depends on that
of its candidate hidden states ˜ ht. The hyperbolic tangent function (tanh) plays a crucial role in
LSTMs and GRUs, restricting the range of (candidate) hidden states, i.e., ˜ ht,ht ∈ (−1,1)dh. The
tanh helps stabilize the training and mitigates vanishing gradients that result from applying sigmoid
(σ) activations to linear transformations of the hidden state (e.g., zt = σ(Lineardh
([xt,ht−1]))). In
the previous step, these hidden state dependencies were removed. As such, we simplify GRU further
by removing the range restriction (tanh) on the (candidate) hidden states as follows:
˜
ht = tanh(Lineardh
(xt)) ⇒ ˜ ht = Lineardh
(xt)
3.1.3 minGRU
Combining the two simplification steps results in a minimal version of GRU (minGRU):
3 Methodology
Interestingly, we can see that the GRU’s hidden state and LSTM’s cell state recurrences resemble the
vector formulation. In this section, we demonstrate that GRUs and LSTMs are trainable via parallel
scan by removing their previous state dependencies from their various gates.
3.669
3 Metodología
Curiosamente, podemos observar que el estado oculto de la GRU y las recurrencias del estado de celda de la LSTM se asemejan a la formulación vectorial. En esta sección, demostramos que las GRU y las LSTM se pueden entrenar mediante escaneo paralelo, eliminando las dependencias de estado previas de sus diversas puertas. Basándonos en esto, simplificamos aún más estas RNN eliminando sus restricciones en el rango de salida (es decir, tanh). Combinando los pasos, describimos versiones mínimas de las GRU y las LSTM (minGRU y minLSTM) que se pueden entrenar en paralelo. 3
3.1 AMinimalGRU:minGRU
3.1.1 Paso 1: Eliminar las dependencias de estado previas de las puertas
Revisando la recurrencia de estado oculta de GRU, que funciona de la siguiente manera:
ht = (1−zt)⊙ht−1 +zt ⊙ ˜ ht
Podemos observar que la recurrencia se asemeja a la formulación del escaneo paralelo mencionada anteriormente: vt =
at ⊙ vt−1 + bt, donde at ← (1 − zt), bt ← zt ⊙ ˜ ht, y vt ← ht. Sin embargo, zt y ˜ ht dependen del estado oculto previo ht−1, es decir, zt = σ(Lineardh
([xt,ht−1])) y ˜ ht = tanh(Lineardh
([xt,rt ⊙ ht−1])). Como resultado, no es posible aplicar el escaneo paralelo tal como está, ya que las entradas del algoritmo a1,...,an y b1,...,bn dependen de que ya se conozcan sus salidas h1,...,hn−1.
Una solución sencilla es simplificar GRU eliminando sus dependencias del estado oculto previo (es decir, ht−1). Específicamente, los cambios son los siguientes:
zt = σ(Lineardh
([xt,ht−1]))
rt = σ(Lineardh
([xt,ht−1]))
˜
ht = tanh(Lineardh
([xt,rt ⊙ ht−1]))
⇒ zt=σ(Lineardh
(xt))
˜
ht = tanh(Lineardh
(xt))
Al eliminar la dependencia de ht−1 del estado oculto candidato ˜ ht, la puerta de reinicio rt que controlaría el peso de ht−1 ya no es necesaria y se elimina. Sin las dependencias de los estados ocultos previos, las entradas del algoritmo a1,...,an y b1,...,bn se calculan fácilmente en paralelo y, por lo tanto, pueden utilizarse para calcular h1,...,hn de manera eficiente mediante el escaneo paralelo. Si bien ha habido inquietudes teóricas sobre la ausencia de dependencias de estados previos (Merrill et al., 2024), también existe evidencia empírica sustancial que respalda la eficacia de los modelos que omiten estas dependencias, como xLSTM (Beck et al., 2024) y Mamba (Gu y Dao, 2024). En lugar de modelar explícitamente las dependencias de estados previos para capturar dependencias de largo alcance, este tipo de modelos recurrentes pueden aprenderlas mediante el apilamiento de múltiples capas. Cabe destacar que, en el artículo sobre xLSTM, su versión totalmente paralelizada (xLSTM[1:0]), que elimina las dependencias de estados ocultas, tuvo un rendimiento similar, e incluso superior, al de las versiones que conservan estas dependencias (p. ej., xLSTM[7:1]). 3.1.2 Paso 2: Restricción del rango de eliminación de estados candidatos
En la recurrencia del estado oculto de GRU, la proporción transferida del estado oculto anterior (1−zt)
y la cantidad añadida para el nuevo estado oculto candidato (zt) suman 1. Como resultado, la escala del valor del estado oculto de GRU es independiente del tiempo. En cambio, la escala de su estado oculto depende de la de sus estados ocultos candidatos ˜ ht. La función tangente hiperbólica (tanh) desempeña un papel crucial en los LSTM y GRU, restringiendo el rango de estados ocultos (candidatos), es decir, ˜ ht, ht ∈ (−1,1)dh. La tanh ayuda a estabilizar el entrenamiento y mitiga los gradientes de desvanecimiento que resultan de la aplicación de activaciones sigmoideas
(σ) a las transformaciones lineales del estado oculto (p. ej., zt = σ(Lineardh
([xt, ht−1]))). En el paso anterior, se eliminaron estas dependencias de estado ocultas. Por lo tanto, simplificamos aún más GRU eliminando la restricción de rango (tanh) en los estados ocultos (candidatos) de la siguiente manera:
˜
ht = tanh(Lineardh
(xt)) ⇒ ˜ ht = Lineardh
(xt)
3.1.3 minGRU
La combinación de los dos pasos de simplificación da como resultado una versión mínima de GRU (minGRU):
Enviar comentarios
Pulsa el tabulador para acceder a las acciones
/*/*/*/**/


Detectar idioma

inglés

español

Francés

español

inglés

Francés
.2 GRU
Simplifying LSTM, Cho et al. (2014) introduced the Gated Recurrent Unit (GRU), which uses only
two gates and a single state (hidden state), in contrast to the LSTM’s three gates and two states
(hidden state and cell state). This reduced complexity allows GRUs to achieve faster training and
inference times while still performing competitively on many tasks. GRUs are computed as follows:
(Hidden State Recurrence)
(Update Gate)
(Reset Gate)
(Candidate Hidden State)
ht = (1−zt)⊙ht−1 +zt ⊙ ˜ ht
zt = σ(Lineardh
([xt,ht−1]))
rt = σ(Lineardh
([xt,ht−1]))
˜
ht = tanh(Lineardh
([xt,rt ⊙ ht−1]))
where ˜ ht represents the candidate hidden state, a potential new value for the hidden state. GRU
combinestheforget and input gates of LSTMintoasingleupdategate, zt ∈ (0,1), whichdetermines
how much of the past information should be carried forward (i.e., 1 − zt) and how much new
information from the candidate hidden state should be added (i.e., zt). Additionally, GRU removes
LSTM’soutput gate and introduces a reset gate rt, which controls how much of the past hidden state
ht−1 is used when computing the candidate hidden state ˜ ht.
By reducing the number of gates and states, GRU also decreases the total number of parameters and
computations, requiring only O(3dh(dx + dh)) parameters. However, both GRUs and LSTMs are
still sequential-only models. As such, they require backpropagation through time (BPTT) during
training, resulting in linear training time and limiting their ability to scale to long contexts.
2.3 Parallel Scan
Due to this limitation, the introduction of Transformers in 2017 revolutionized the field by replacing
LSTMs and GRUs as the de facto method for sequence modelling. Transformers leverage par
allelization during training, overcoming the sequential bottleneck of traditional recurrent models.
However, instead, Transformers have a quadratic complexity with respect to the sequence length,
limiting their ability to scale to very long contexts, especially in resource-constrained settings.
In response, a resurgence of new recurrent sequence models has emerged, offering alternatives to
Transformers. These models achieve comparable performance while being trainable in parallel and
avoid the backpropagation through time (BPTT) issues that plagued traditional RNNs (e.g., LSTMs
and GRUs). Among these innovations, many architectures rely on the parallel prefix scan algo
rithm (Blelloch, 1990) for efficient training.
The parallel scan algorithm is a parallel computation method for computing N prefix computations
from N sequential data points via an associative operator ⊕ (e.g., + and ×). The algorithm effi
ciently computes the sequence of prefix sums { k
i=1 ui}N
k=1 from the input sequence {uk}N
k=1. One
important application of the parallel scan algorithm is in computing a popular class of recurrence
relations of the form vt = atvt−1+bt, where vt, at, and bt are real numbers and v0 ← b0 (Martin &
Cundy,2018). Thismethodtakesasinput thesequences a1,...,an andb0,b1,...,bn, andcomputes
the sequence v1,...,vn in parallel. This approach naturally extends to vector-valued recurrences,
such as vt = at ⊙vt−1 +bt, where ⊙ denotes element-wise multiplication.
3.201
.2 GRU
Simplificando el LSTM, Cho et al. (2014) introdujeron la Unidad Recurrente Cerrada (GRU), que utiliza solo dos puertas y un solo estado (estado oculto), a diferencia de las tres puertas y dos estados del LSTM (estado oculto y estado de celda). Esta menor complejidad permite a las GRU lograr tiempos de entrenamiento e inferencia más rápidos, manteniendo un rendimiento competitivo en muchas tareas. Las GRU se calculan de la siguiente manera:
(Recurrencia del estado oculto)
(Puerta de actualización)
(Puerta de reinicio)
(Estado oculto candidato)
ht = (1−zt)⊙ht−1 +zt ⊙ ˜ ht
zt = σ(Lineardh
([xt,ht−1]))
rt = σ(Lineardh
([xt,ht−1]))
˜
ht = tanh(Lineardh
([xt,rt ⊙ ht−1]))
donde ˜ ht representa el estado oculto candidato, un nuevo valor potencial para el estado oculto. GRU
combina las puertas de olvido y de entrada de LSTM en una única puerta de actualización, zt ∈ (0,1), que determina cuánta información pasada debe transferirse (es decir, 1 − zt) y cuánta información nueva del estado oculto candidato debe añadirse (es decir, zt). Además, GRU elimina la puerta de salida de LSTM e introduce una puerta de reinicio rt, que controla la cantidad del estado oculto pasado ht−1 que se utiliza al calcular el estado oculto candidato ˜ ht.
Al reducir el número de puertas y estados, GRU también disminuye el número total de parámetros y cálculos, requiriendo solo O(3dh(dx + dh)) parámetros. Sin embargo, tanto GRU como LSTM siguen siendo modelos exclusivamente secuenciales. Por lo tanto, requieren retropropagación a través del tiempo (BPTT) durante el entrenamiento, lo que resulta en un tiempo de entrenamiento lineal y limita su capacidad de escalar a contextos largos. 2.3 Escaneo Paralelo
Debido a esta limitación, la introducción de los Transformadores en 2017 revolucionó el campo al reemplazar a los LSTM y las GRU como el método de facto para el modelado de secuencias. Los Transformadores aprovechan la paralelización durante el entrenamiento, superando el cuello de botella secuencial de los modelos recurrentes tradicionales.
Sin embargo, los Transformadores presentan una complejidad cuadrática con respecto a la longitud de la secuencia, lo que limita su capacidad de escalar a contextos muy largos, especialmente en entornos con recursos limitados.
En respuesta, ha surgido un resurgimiento de nuevos modelos de secuencias recurrentes que ofrecen alternativas a los Transformadores. Estos modelos logran un rendimiento comparable, a la vez que se pueden entrenar en paralelo y evitan los problemas de retropropagación en el tiempo (BPTT) que afectaban a las RNN tradicionales (por ejemplo, los LSTM y las GRU). Entre estas innovaciones, muchas arquitecturas se basan en el algoritmo de escaneo de prefijo paralelo (Blelloch, 1990) para un entrenamiento eficiente. El algoritmo de escaneo paralelo es un método de cálculo paralelo para calcular N cálculos de prefijos a partir de N puntos de datos secuenciales mediante un operador asociativo ⊕ (p. ej., + y ×). El algoritmo calcula eficientemente la secuencia de sumas de prefijos { k
i=1 ui}N
k=1 a partir de la secuencia de entrada {uk}N
k=1. Una aplicación importante del algoritmo de escaneo paralelo es el cálculo de una clase popular de relaciones de recurrencia de la forma vt = atvt−1+bt, donde vt, at y bt son números reales y v0 ← b0 (Martin y Cundy, 2018). Este método toma como entrada las secuencias a1,..., an y b0, b1,..., bn, y calcula la secuencia v1,..., vn en paralelo. Este enfoque se extiende naturalmente a recurrencias con valores vectoriales, como vt = at ⊙vt−1 +bt, donde ⊙ denota multiplicación elemento por elemento.
Enviar comentarios
Resultados de traducción disponibles
/*/*/*/*


Detectar idioma

inglés

español

Francés

español

inglés

Francés
Transformer efficiency, exploring ideas such as sparsity (Kitaev et al., 2019), low-rank approxima
tions (Wang et al., 2020), and tiling (Dao et al., 2022).
Recently, the scalability limitations of Transformers have sparked renewed interest in alternative ap
proaches: novel recurrent models that are parallelizable and scale more efficiently. Several promis
ing methods have emerged in this space, including state-space models (Gu et al., 2021), linearized
attention (Peng et al., 2023), and more recently, linear recurrent neural networks (Orvieto et al.,
2023). Notably, these state-of-the-art recurrent models leverage input-dependent transitions and
demonstrate strong performance similar to Transformers. These methods have shown success not
only in scaling to large language models but also in extending to other domains, such as image (Zhu
et al., 2024a) and graph-based data (Wang et al., 2024a).
In this work, we revisit sequence modelling from a historical perspective, focusing on the RNNs
that dominated the field for two decades before the rise of Transformers. Specifically, we explore
LSTMs (1997) and GRUs (2014), which are early examples of input-dependent recurrent models.
We show that by removing the dependencies of their gates on previous states, we can train these
models in parallel. Further simplification leads to minimal versions (minLSTMs and minGRUs)
that (1) use fewer parameters than their traditional counterparts, (2) are fully parallelizable during
training, and (3) achieve surprisingly competitive performance on a range of tasks despite their
simplicity, challenging the prevailing trend in the community toward increasing architectural and
algorithmic complexity. In the appendix, we provide implementations of minGRU and minLSTM in
plain PyTorch, with just a few lines of code, making these models lightweight and highly adaptable
for beginners, practitioners, and researchers.
2 Background
In this section, we review traditional recurrent neural networks (RNNs). RNNs are sequence models
that maintain a hidden state across time steps, capturing temporal dependencies. As such, they are
particularly well-suited for tasks involving sequential data, such as time series forecasting, natural
language processing, and other tasks where context from previous steps informs current predictions.
However, vanilla RNNs (Elman, 1990) face challenges related to vanishing and exploding gradients,
which limit their ability to learn long-term dependencies.
2.1 LSTM
To address these issues, Hochreiter & Schmidhuber (1997) introduced Long Short-Term Memory
(LSTM) networks. LSTMs are a highly successful type of RNN specifically designed to mitigate
the vanishing gradient problem, enabling the model to effectively capture long-term dependencies.
LSTMs are computed as follows:
(Hidden State)
(Output Gate)
(Cell State Recurrence)
(Forget Gate)
(Input Gate)
(Candidate Cell State)
ht = ot ⊙tanh(ct)
ot = σ(Lineardh
([xt,ht−1]))
ct = ft ⊙ct−1 +it ⊙ ˜ct
ft = σ(Lineardh
([xt,ht−1]))
it = σ(Lineardh
([xt,ht−1]))
˜
ct = tanh(Lineardh
([xt,ht−1]))
where ⊙ denotes element-wise multiplication of vectors, t is the current timestep, and ht is the
outputted hidden state. [xt,ht−1] represents the concatenation of the input vector xt at time step t
with the previous hidden state ht−1. dh denotes the size of the hidden state, while ct is the cell state,
which carries information across time steps, and ˜ct is the candidate cell state that will be added to
the cell state.
The gates it, ft, and ot control the flow of information through the LSTM. The input gate it deter
mines how much new information from the candidate cell state ˜ct should be added to the cell state
ct. The forget gate ft determines what portion of the previous cell state ct−1 should be discarded.
The output gate ot determines what information from the cell state should be output as the hidden
state ht. The functions σ (sigmoid) and tanh are used for scaling the values, ensuring that the out
2
puts do not explode or vanish during training. An LSTM module maintains both a cell state and a
hidden state, and, in total, contains O(4dh(dx + dh)) parameters, where dx is the input size.
4.186
Eficiencia de los transformadores, explorando ideas como la escasez (Kitaev et al., 2019), las aproximaciones de bajo rango (Wang et al., 2020) y el teselado (Dao et al., 2022).
Recientemente, las limitaciones de escalabilidad de los transformadores han despertado un renovado interés en enfoques alternativos: nuevos modelos recurrentes que son paralelizables y escalan de forma más eficiente. Han surgido varios métodos prometedores en este ámbito, incluyendo modelos de espacio de estados (Gu et al., 2021), atención linealizada (Peng et al., 2023) y, más recientemente, redes neuronales recurrentes lineales (Orvieto et al., 2023). Cabe destacar que estos modelos recurrentes de vanguardia aprovechan las transiciones dependientes de la entrada y demuestran un rendimiento sólido similar al de los transformadores. Estos métodos han demostrado ser eficaces no solo al escalar a grandes modelos lingüísticos, sino también al extenderse a otros dominios, como imágenes (Zhu et al., 2024a) y datos basados ​​en grafos (Wang et al., 2024a).
En este trabajo, revisamos el modelado de secuencias desde una perspectiva histórica, centrándonos en las RNN que dominaron el campo durante dos décadas antes del auge de los Transformers. Específicamente, exploramos los LSTM (1997) y las GRU (2014), ejemplos tempranos de modelos recurrentes dependientes de la entrada.
Mostramos que, al eliminar las dependencias de sus puertas con respecto a estados previos, podemos entrenar estos modelos en paralelo. Una mayor simplificación da como resultado versiones mínimas (minLSTM y minGRU) que (1) utilizan menos parámetros que sus contrapartes tradicionales, (2) son totalmente paralelizables durante el entrenamiento y (3) logran un rendimiento sorprendentemente competitivo en diversas tareas a pesar de su simplicidad, desafiando la tendencia predominante en la comunidad hacia una mayor complejidad arquitectónica y algorítmica. En el apéndice, proporcionamos implementaciones de minGRU y minLSTM en PyTorch simple, con solo unas pocas líneas de código, lo que hace que estos modelos sean ligeros y altamente adaptables para principiantes, profesionales e investigadores.

2 Antecedentes
En esta sección, revisamos las redes neuronales recurrentes (RNN) tradicionales. Las RNN son modelos de secuencia que mantienen un estado oculto a lo largo del tiempo, capturando dependencias temporales. Por lo tanto, son particularmente adecuadas para tareas que involucran datos secuenciales, como la predicción de series temporales, el procesamiento del lenguaje natural y otras tareas donde el contexto de los pasos anteriores informa las predicciones actuales. Sin embargo, las RNN tradicionales (Elman, 1990) enfrentan desafíos relacionados con gradientes evanescentes y explosivos, lo que limita su capacidad para aprender dependencias a largo plazo.
2.1 LSTM
Para abordar estos problemas, Hochreiter y Schmidhuber (1997) introdujeron las redes de memoria a corto y largo plazo (LSTM). Las LSTM son un tipo de RNN muy exitoso, diseñado específicamente para mitigar el problema del gradiente evanescente, lo que permite que el modelo capture eficazmente las dependencias a largo plazo. Los LSTM se calculan de la siguiente manera:
(Estado oculto)
(Puerta de salida)
(Recurrencia del estado de la celda)
(Puerta de olvido)
(Puerta de entrada)
(Estado de la celda candidata)
ht = ot ⊙tanh(ct)
ot = σ(Lineardh
([xt,ht−1]))
ct = ft ⊙ct−1 +it ⊙ ˜ct
ft = σ(Lineardh
([xt,ht−1]))
it = σ(Lineardh
([xt,ht−1]))
˜
ct = tanh(Lineardh
([xt,ht−1]))
donde ⊙ denota la multiplicación de vectores elemento por elemento, t es el paso de tiempo actual y ht es el estado oculto resultante. [xt,ht−1] representa la concatenación del vector de entrada xt en el paso de tiempo t con el estado oculto previo ht−1. dh denota el tamaño del estado oculto, mientras que ct es el estado de la celda, que transporta información a través de los pasos de tiempo, y ˜ct es el estado de celda candidato que se añadirá al estado de celda.
Las puertas it, ft y ot controlan el flujo de información a través del LSTM. La puerta de entrada it determina cuánta información nueva del estado de celda candidato ˜ct debe añadirse al estado de celda ct. La puerta de olvido ft determina qué porción del estado de celda previo ct−1 debe descartarse.
La puerta de salida ot determina qué información del estado de celda debe emitirse como el estado oculto ht. Las funciones σ (sigmoide) y tanh se utilizan para escalar los valores, lo que garantiza que las salidas no exploten ni desaparezcan durante el entrenamiento. Un módulo LSTM mantiene tanto un estado de celda como un estado oculto y, en total, contiene O(4dh(dx + dh)) parámetros, donde dx es el tamaño de entrada.
Enviar comentarios
Resultados de traducción disponibles
/*/*/*


Detectar idioma

inglés

español

Francés

español

inglés

Francés
I have been working on a project (a game to be specific) and I feel that I should start over with different libraries. So when doing this I reinstalled Code::Blocks and setup my new libraries and includes.

But as of now Im having a problem starting u[ my new project to test if all of the includes work. This problem is: libstdc++-6.dll was not found. At first i wondered if I could just find this file online, but its nowhere to be found(or at least the many places I have searched...) Soon after, I tried loading up my old project, and the same problem happened again(wierd... ._.) I was thinking its maybe my compiler, so I used my older compiler and it did the same thing! At this moment I held the problem off for tomorrow(which is today)

So my question is: If anyone else had this problem, how would you solve it?

Im using Code::Blocks with MinGW as the compiler on Windows Vista 32 bit.

*****EDIT*****

Here are the Build options in my project. Note that these are the settings in the Project, not the global compiler:

In (project name)->Compiler settings->Otehr options:

(I use // to seperate the commands)

-mthreads//
-fmessage-length=0//
-fexceptions//
-fident//
In (project name)->Compiler settings->#define:

WIN32//
_WINDOWS//
In (project name)->Linker settings->Other linker options:

-static-libstdc++//
-static-libgcc//
-Wl,--enable-auto-image-base//
-Wl,--add-stdcall-alias//
-Wl,--enable-auto-import//
In linker->link libraries i have various links to files with a .a extension, these files include Bullet PHysics, Ogre3D, and SFML

In the search directories i have links to the MinGW/bin, and the MinGW/lib directories, along with other links to different libraries.

My Compiler is MinGW, a GNU GCC compiler for windows 32 bit. and the IDE is Codeblocks. Also note that in Debug and Release settings on the project, there is nothing.

Most of these setings are also pieces that i got from the Ogre3D Application setup tutorial if that is of any help.

c++mingwlibraries
1.996
He estado trabajando en un proyecto (un juego, para ser más precisos) y creo que debería empezar de cero con bibliotecas diferentes. Así que, al hacerlo, reinstalé Code::Blocks y configuré mis nuevas bibliotecas e inclusiones.

Pero ahora mismo tengo un problema al iniciar mi nuevo proyecto para comprobar si todas las inclusiones funcionan. El problema es que no se encontró libstdc++-6.dll. Al principio me pregunté si podría encontrar este archivo en línea, pero no lo encuentro por ningún lado (o al menos en los muchos sitios que he buscado...). Poco después, intenté cargar mi antiguo proyecto y el mismo problema volvió a ocurrir (¡qué raro!). Pensé que quizás fuera mi compilador, así que usé mi compilador anterior y ¡hizo lo mismo! Por el momento, dejé el problema para mañana (que es hoy).

Mi pregunta es: si alguien más tuviera este problema, ¿cómo lo solucionaría?

Estoy usando Code::Blocks con MinGW como compilador en Windows Vista de 32 bits.

*****EDITAR*****

Aquí están las opciones de compilación de mi proyecto. Tenga en cuenta que estas son las configuraciones del proyecto, no del compilador global:

En (nombre del proyecto) -> Configuración del compilador -> Otras opciones:

(Uso // para separar los comandos)

-mthreads//
-fmessage-length=0//
-fexceptions//
-fident//
En (nombre del proyecto) -> Configuración del compilador -> #define:

WIN32//
_WINDOWS//
En (nombre del proyecto) -> Configuración del enlazador -> Otras opciones del enlazador:

-static-libstdc++//
-static-libgcc//
-Wl,--enable-auto-image-base//
-Wl,--add-stdcall-alias//
-Wl,--enable-auto-import//
En enlazador -> enlazar bibliotecas, tengo varios enlaces a archivos con extensión .a, como Bullet PHysics, Ogre3D y SFML

En los directorios de búsqueda, tengo enlaces a los directorios MinGW/bin y MinGW/lib, junto con otros enlaces a diferentes bibliotecas.

Mi compilador es MinGW, un compilador GNU GCC para Windows de 32 bits, y el IDE es Codeblocks. Tenga en cuenta también que en la configuración de depuración y lanzamiento del proyecto no hay nada.

La mayoría de estas configuraciones también provienen del tutorial de configuración de la aplicación Ogre3D, por si le sirve de ayuda.

c++mingwlibraries
Enviar comentarios
Resultados de traducción disponibles
/*/*/*



Detectar idioma

inglés

español

Francés

español

inglés

Francés
Combining the three steps results in a minimal version of LSTM (minLSTM):
LSTM
ht = ot ⊙tanh(ct)
ot = σ(Lineardh
([xt,ht−1]))
ct = ft ⊙ct−1 +it ⊙ ˜ct
ft = σ(Lineardh
([xt,ht−1]))
it = σ(Lineardh
([xt,ht−1]))
˜
ct = tanh(Lineardh
([xt,ht−1]))
⇒
minLSTM
ht = ft ⊙ht−1 +it ⊙ ˜ ht
ft = σ(Lineardh
(xt))
it = σ(Lineardh
(xt))
˜
ht = Lineardh
(xt)
where time-independent outputs can be achieved using a hidden state recurrence ht = f′
t ⊙ht−1 +
i′
t ⊙ ˜ ht with normalized forget f′
t and input it gates computed as f′
t,i′
t ← ft
ft+it 
, it
ft+it 
.
The resulting model is significantly more efficient than the original LSTM, requiring only O(3dhdx)
parameters compared to LSTM’s O(4dh(dx + dh)). Considering state expansion (i.e., dh = αdx,
where α ≥ 1),minLSTMusesapproximately 38%,25%,19%,or15%oftheparameters of a LSTM
when α = 1,2,3, or 4 respectively.
Additionally, the minimal version of LSTM can now be trained in parallel using the parallel scan
algorithm, bypassing the need for backpropagation through time (BPTT). Pseudocode and a simple
PyTorch implementation are included in the Appendix.
4 WereRNNsAllWeNeeded?
In this section, we compare the minimal versions (minLSTMs and minGRUs) with their traditional
counterparts (LSTMs and GRUs) and modern sequence models. Pseudocode, PyTorch implementa
tion, and detailed information regarding the experiment setup are available in the Appendix.
1.398
La combinación de los tres pasos da como resultado una versión mínima de LSTM (minLSTM):
LSTM
ht = ot ⊙tanh(ct)
ot = σ(Lineardh
([xt,ht−1]))
ct = ft ⊙ct−1 +it ⊙ ˜ct
ft = σ(Lineardh
([xt,ht−1]))
it = σ(Lineardh
([xt,ht−1]))
˜
ct = tanh(Lineardh
([xt,ht−1]))
⇒
minLSTM
ht = ft ⊙ht−1 +it ⊙ ˜ ht
ft = σ(Lineardh
(xt))
it = σ(Lineardh
(xt))
˜
ht = Lineardh
(xt)
donde Se pueden obtener salidas independientes del tiempo utilizando una recurrencia de estado oculta ht = f′
t ⊙ht−1 +
i′
t ⊙ ˜ ht con f′
t de olvido normalizado y puertas it de entrada calculadas como f′
t,i′
t ← ft
ft+it
, it
ft+it
.
El modelo resultante es significativamente más eficiente que el LSTM original, requiriendo solo O(3dhdx) parámetros en comparación con los O(4dh(dx + dh) del LSTM. Considerando la expansión de estado (es decir, dh = αdx, donde α ≥ 1), minLSTM utiliza aproximadamente el 38 %, 25 %, 19 % o 15 % de los parámetros de un LSTM cuando α = 1, 2, 3 o 4, respectivamente. Además, la versión mínima de LSTM ahora puede entrenarse en paralelo mediante el algoritmo de escaneo paralelo, lo que elimina la necesidad de retropropagación en el tiempo (BPTT). El pseudocódigo y una implementación sencilla de PyTorch se incluyen en el Apéndice.

4 ¿Eran las RNN todo lo que necesitábamos?

En esta sección, comparamos las versiones mínimas (minLSTM y minGRU) con sus contrapartes tradicionales (LSTM y GRU) y con los modelos de secuencia modernos. El pseudocódigo, la implementación de PyTorch e información detallada sobre la configuración del experimento están disponibles en el Apéndice.
Enviar comentarios
Pulsa el tabulador para acceder a las acciones
/*/*/*/*


Detectar idioma

inglés

español

Francés

español

inglés

Francés
RWKV-7"Goose"withExpressiveDynamicStateEvolution
BoPeng1,2,∗ ,Ruichong Zhang3,*, Daniel Goldstein2,4,*,
Eric Alcaide2,5, Xingjian Du6, HaowenHou7, Jiaju Lin8, Jiaxing Liu9, JannaLu4,10, William
Merrill11, Guangyu Song2,12, Kaifeng Tan13, Saiteja Utpala2, Nathan Wilce2,4, Johan S. Wind14,
Tianyi Wu15, DanielWuttke2,16, and Christian Zhou-Zheng2
arXiv:2503.14456v2  [cs.CL]  30 Mar 2025
1RWKVProject (under Linux Foundation AI & Data), 2EleutherAI, 3Tsinghua University, 4Recursal
AI, 5Dalle Molle Institute for Artificial Intelligence USI-SUPSI, 6University of Rochester,
7GuangdongLaboratory of Artificial Intelligence and Digital Economy (SZ), 8Pennsylvania State
University, 9Zhejiang University, 10George Mason University, 11New York University, 12Tano Labs,
13Shenzhen University, 14University of Oslo, 15Beijing Normal University, 16Denigma
Abstract
We present RWKV-7 "Goose", a new sequence modeling architecture with constant memory
usage andconstant inference time per token. Despite being trained on dramatically fewer tokens
than other top models, our 2.9 billion parameter language model achieves a new 3B SoTA on
multilingualtasksandmatchesthecurrent3BSoTAonEnglishlanguagedownstreamperformance.
RWKV-7introduces anewlygeneralized formulation of the delta rule with vector-valued gating
andin-context learning rates, as well as a relaxed value replacement rule. We show that RWKV-7
can perform state tracking and recognize all regular languages, while retaining parallelizability of
training. This exceeds the capabilities of Transformers under standard complexity conjectures,
whicharelimited to TC0. TodemonstrateRWKV-7’slanguagemodelingcapability, wealsopresent
anextended opensource 3.1 trillion token multilingual corpus, and train four RWKV-7 models
ranging from 0.19 billion to 2.9 billion parameters on this dataset.
To foster openness, reproduction, and adoption, we release our models1 and dataset component
listing2 on Hugging Face, and our training and inference code3 on GitHub; all under the Apache
2.026
RWKV-7 "Ganso" con Evolución de Estado Dinámico Expresivo
BoPeng1,2,∗, Ruichong Zhang3,*, Daniel Goldstein2,4,*,
Eric Alcaide2,5, Xingjian Du6, HaowenHou7, Jiaju Lin8, Jiaxing Liu9, JannaLu4,10, William
Merrill11, Guangyu Song2,12, Kaifeng Tan13, Saiteja Utpala2, Nathan Wilce2,4, Johan S. Wind14,
Tianyi Wu15, DanielWuttke2,16 y Christian Zhou-Zheng2
arXiv:2503.14456v2 [cs.CL] 30 de marzo de 2025
1. Proyecto RWKV (bajo la Fundación Linux de IA y Datos), 2. EleutherAI, 3. Universidad de Tsinghua, 4. Recursal
IA, 5. Instituto Dalle Molle de Inteligencia Artificial USI-SUPSI, 6Universidad de Rochester,
7Laboratorio de Inteligencia Artificial y Economía Digital de Guangdong (SZ), 8Universidad Estatal de Pensilvania, 9Universidad de Zhejiang, 10Universidad George Mason, 11Universidad de Nueva York, 12Tano Labs,
13Universidad de Shenzhen, 14Universidad de Oslo, 15Universidad Normal de Pekín, 16Denigma
Resumen
Presentamos RWKV-7 "Goose", una nueva arquitectura de modelado de secuencias con uso de memoria constante y tiempo de inferencia constante por token. A pesar de haber sido entrenado con una cantidad considerablemente menor de tokens que otros modelos de alto nivel, nuestro modelo de lenguaje de 2900 millones de parámetros alcanza un nuevo SoTA de 3 000 millones en tareas multilingües e iguala el SoTA actual de 3 000 millones en rendimiento posterior en inglés. RWKV-7 introduce una formulación recientemente generalizada de la regla delta con puertas con valores vectoriales y tasas de aprendizaje en contexto, así como una regla de reemplazo de valores relajada. Demostramos que RWKV-7 puede realizar seguimiento de estados y reconocer todos los lenguajes regulares, manteniendo la paralelización del entrenamiento. Esto supera las capacidades de los transformadores bajo conjeturas de complejidad estándar, limitadas a TC0. Para demostrar la capacidad de modelado de lenguaje de RWKV-7, también presentamos un corpus multilingüe de código abierto extendido de 3,1 billones de tokens y entrenamos cuatro modelos RWKV-7 que abarcan desde 190 millones hasta 2900 millones de parámetros en este conjunto de datos.
Para fomentar la apertura, la reproducción y la adopción, publicamos nuestros modelos¹ y el listado de componentes del conjunto de datos² en Hugging Face, y nuestro código de entrenamiento e inferencia³ en GitHub; todo ello bajo Apache.
Enviar comentarios
Resultados de traducción disponibles
//*/*


Detectar idioma

inglés

español

Francés

español

inglés

Francés
ideas of the Long Short-Term Memory (LSTM). Since then, LSTMs have stood
the test of time and contributed to numerous deep learning success stories, in
particular they constituted the first Large Language Models (LLMs). However,
the advent of the Transformer technology with parallelizable self-attention at its
core marked the dawn of a new era, outpacing LSTMs at scale. We now raise a
simple question: How far do we get in language modeling when scaling LSTMs to
billions of parameters, leveraging the latest techniques from modern LLMs, but
mitigating known limitations of LSTMs? Firstly, we introduce exponential gating
with appropriate normalization and stabilization techniques. Secondly, we modify
the LSTM memory structure, obtaining: (i) sLSTM with a scalar memory, a scalar
update, and new memory mixing, (ii) mLSTM that is fully parallelizable with a
matrix memory and a covariance update rule. Integrating these LSTM extensions
into residual block backbones yields xLSTM blocks that are then residually stacked
into xLSTM architectures. Exponential gating and modified memory structures
boost xLSTM capabilities to perform favorably when compared to state-of-the-art
Transformers and State Space Models, both in performance and scaling.
Code available at: https://github.com/NX-AI/xlst
1.298
Ideas de la Memoria a Largo Plazo (MLTP). Desde entonces, los MLTP han resistido el paso del tiempo y han contribuido a numerosos casos de éxito en el aprendizaje profundo; en particular, constituyeron los primeros Modelos de Lenguaje Grandes (MLG). Sin embargo, la llegada de la tecnología Transformer, con autoatención paralelizable como núcleo, marcó el inicio de una nueva era, superando a los MLTP a escala. Ahora planteamos una pregunta sencilla: ¿Hasta dónde llegamos en el modelado del lenguaje al escalar los MLTP a miles de millones de parámetros, aprovechando las técnicas más recientes de los MLG modernos, pero mitigando las limitaciones conocidas de los MLTP? En primer lugar, introducimos la activación exponencial con técnicas adecuadas de normalización y estabilización. En segundo lugar, modificamos la estructura de memoria del MLTP, obteniendo: (i) un MLTP con una memoria escalar, una actualización escalar y una nueva mezcla de memoria, (ii) un MLTP totalmente paralelizable con una memoria matricial y una regla de actualización de covarianza. La integración de estas extensiones LSTM en las estructuras troncales de bloques residuales genera bloques xLSTM que se apilan residualmente en arquitecturas xLSTM. La activación exponencial y las estructuras de memoria modificadas mejoran las capacidades de xLSTM para un rendimiento favorable en comparación con los transformadores y modelos de espacio de estados de última generación, tanto en rendimiento como en escalabilidad.
Código disponible en: https://github.com/NX-AI/xlst
Enviar comentarios
Resultados de traducción disponibles 
//*/**/*/


Detectar idioma

inglés

español

Francés

español

inglés

Francés
cell with constant error carousel and gating. 2. New sLSTM and mLSTM memory cells that introduce
exponential gating. sLSTM offers a new memory mixing technique. mLSTM is fully parallelizable
with a novel matrix memory cell state and new covariance update rule. 3. mLSTM and sLSTM in
residual blocks yield xLSTM blocks. 4. Stacked xLSTM blocks give an xLSTM architecture.
1 Introduction
The Long Short-Term Memory (LSTM) ideas (Hochreiter, 1991; Hochreiter & Schmidhuber,
1997b,a), i.e., the constant error carousel and gating, were introduced to overcome the vanishing
gradient problem of recurrent neural networks (Hochreiter, 1991; Hochreiter et al., 2000):
ct =
ft
ct−1 +
it
zt , ht =
ot ψ(
ct ) .
(1)
The constant error carousel is the additive update of the cell state ct−1 (green) by cell inputs zt and
moderated by sigmoid gates (blue). The input gate it and the forget gate ft control this update, while
the output gate ot controls the output of the memory cell, i.e. the hidden state ht. The cell state is
normalized or squashed by ψ and then output gating gives the hidden state.
1.089
Celda con carrusel de errores constantes y puertas. 2. Nuevas celdas de memoria sLSTM y mLSTM que introducen puertas exponenciales. sLSTM ofrece una nueva técnica de mezcla de memoria. mLSTM es totalmente paralelizable con un novedoso estado de celda de memoria matricial y una nueva regla de actualización de covarianza. 3. mLSTM y sLSTM en bloques residuales generan bloques xLSTM. 4. Los bloques xLSTM apilados proporcionan una arquitectura xLSTM. 1 Introducción
Las ideas de la memoria a largo plazo (MLTP) (Hochreiter, 1991; Hochreiter y Schmidhuber, 1997b,a), es decir, el carrusel de errores constantes y las puertas, se introdujeron para superar el problema del gradiente de desaparición de las redes neuronales recurrentes (Hochreiter, 1991; Hochreiter et al., 2000):
ct =
ft
ct−1 +
it
zt, ht =
ot ψ(
ct).
(1)
El carrusel de errores constantes es la actualización aditiva del estado de la celda ct−1 (verde) por las entradas de la celda zt y moderada por las puertas sigmoideas (azul). La puerta de entrada it y la puerta de olvido ft controlan esta actualización, mientras que la puerta de salida ot controla la salida de la celda de memoria, es decir, el estado oculto ht. El estado de la celda se normaliza o se reduce mediante ψ y luego la puerta de salida proporciona el estado oculto.
Enviar comentarios
Resultados de traducción disponibles
//*//*/*


Detectar idioma

inglés

español

Francés

español

inglés

Francés
Despite their tremendous successes,
LSTMs have three main limitations:
(i) Inability to revise storage deci
sions. We exemplify this limitation
via the Nearest Neighbor Search prob
lem (see also Appendix B): With a ref
erence vector given, a sequence must
be scanned sequentially for the most
similar vector in order to provide its
attached value at sequence end. The
left panel of Figure 2 shows the mean
squared error at this task. LSTM strug
gles to revise a stored value when a
more similar vector is found, while
our new xLSTM remediates this limi
tation by exponential gating. (ii) Lim
ited storage capacities, i.e., informa
tion must be compressed into scalar
cell states. We exemplify this limita
tion via Rare Token Prediction. In the
right panel of Figure 2, the perplex
ity of token prediction on Wikitext
103 (Merity et al., 2017) is given for
partitions of different token frequency.
Figure 2: LSTM limitations. Left: Nearest Neighbor Search
problem in terms of mean squared error (MSE). Given a
reference vector, a sequence is scanned sequentially for the
most similar vector with the objective to return its attached
value at sequence end. LSTM struggles to revise a stored
value when a more similar vector is found. Our new xLSTM
overcomes this limitation by exponential gating. Right: Rare
Token Prediction. The perplexity (PPL) of token prediction
on Wikitext-103, in partitions of token frequency. LSTM
performs worse on predicting rare tokens because of its lim
ited storage capacities, whereas our new xLSTM solves this
problem via a matrix memory.
LSTM performs worse on rare tokens because of its limited storage capacities. Our new xLSTM
solves this problem by a matrix memory. (iii) Lack of parallelizability due to memory mixing, i.e.,
the hidden-hidden connections between hidden states from one time step to the next, which enforce
1.859
A pesar de sus enormes éxitos, los LSTM presentan tres limitaciones principales: (i) Imposibilidad de revisar las decisiones de almacenamiento. Ejemplificamos esta limitación mediante el problema de búsqueda del vecino más cercano (véase también el Apéndice B): Con un vector de referencia dado, una secuencia debe escanearse secuencialmente en busca del vector más similar para obtener su valor asociado al final de la secuencia. El panel izquierdo de la Figura 2 muestra el error cuadrático medio en esta tarea. El LSTM tiene dificultades para revisar un valor almacenado cuando se encuentra un vector más similar, mientras que nuestro nuevo xLSTM soluciona esta limitación mediante una puerta exponencial. (ii) Capacidades de almacenamiento limitadas, es decir, la información debe comprimirse en estados de celda escalares. Ejemplificamos esta limitación mediante la predicción de tokens raros. En el panel derecho de la Figura 2, se presenta la perplejidad de la predicción de tokens en Wikitext-103 (Merity et al., 2017) para particiones con diferente frecuencia de tokens.
Figura 2: Limitaciones de LSTM. Izquierda: Problema de búsqueda del vecino más cercano en términos de error cuadrático medio (MSE). Dado un vector de referencia, se escanea secuencialmente una secuencia en busca del vector más similar con el objetivo de devolver su valor asociado al final de la secuencia. LSTM tiene dificultades para revisar un valor almacenado cuando se encuentra un vector más similar. Nuestro nuevo xLSTM supera esta limitación mediante una puerta exponencial. Derecha: Predicción de tokens raros. La perplejidad (PPL) de la predicción de tokens en Wikitext-103, en particiones con diferente frecuencia de tokens. LSTM tiene un peor rendimiento en la predicción de tokens raros debido a su limitada capacidad de almacenamiento, mientras que nuestro nuevo xLSTM resuelve este problema mediante una memoria matricial. LSTM tiene un peor rendimiento con tokens raros debido a su limitada capacidad de almacenamiento. Nuestro nuevo xLSTM resuelve este problema mediante una memoria matricial. (iii) Falta de paralelización debido a la mezcla de memoria, es decir, las conexiones ocultas entre estados ocultos de un paso de tiempo al siguiente, que refuerzan
Enviar comentarios
Resultados de traducción disponibles 
/*/*/**/*/


Detectar idioma

inglés

español

Francés

español

inglés

Francés
2 Extended Long Short-Term Memory
To overcome the LSTM limitations, Extended Long Short-Term Memory (xLSTM) introduces two
main modifications to the LSTM idea of Equation (1). Those modifications — exponential gating
and novel memory structures — enrich the LSTM family by two members: (i) the new sLSTM (see
Section 2.2) with a scalar memory, a scalar update, and memory mixing, and (ii) the new mLSTM
(see Section 2.3) with a matrix memory and a covariance (outer product) update rule, which is fully
parallelizable. Both sLSTM and mLSTM enhance the LSTM through exponential gating. To enable
parallelization, the mLSTM abandons memory mixing, i.e., the hidden-hidden recurrent connections.
Both mLSTMandsLSTMcanbeextended to multiple memory cells, where sLSTM features memory
mixing across cells. Further, the sLSTM can have multiple heads without memory mixing across the
heads, but only memory mixing across cells within each head. This introduction of heads for sLSTM
together with exponential gating establishes a new way of memory mixing. For mLSTM multiple
heads and multiple cells are equivalent.
Integrating these new LSTM variants into residual block modules results in xLSTM blocks (see
Section 2.4). Residually stacking those xLSTM blocks in architectures provides xLSTM architectures
(see Section 2.4). See Figure 1 for the xLSTM architecture with its components.
2.1 Review of the Long Short-Term Memory
The original LSTM idea (Hochreiter, 1991; Hochreiter & Schmidhuber, 1997b,a) introduced the
scalar memory cell as a central processing and storage unit that avoids vanishing gradients (Hochreiter,
1991; Hochreiter et al., 2000) through the constant error carousel (the cell state update). The memory
cell contains three gates: input, output, and forget gate. The forget gate has been introduced by Gers
et al. (2000). The update rules of the LSTM memory cell at time step t are
1.897
2 Memoria Extendida a Largo y Corto Plazo
Para superar las limitaciones de la LSTM, la Memoria Extendida a Largo y Corto Plazo (xLSTM) introduce dos modificaciones principales al concepto de LSTM de la Ecuación (1). Estas modificaciones —puerta exponencial
y nuevas estructuras de memoria— enriquecen la familia LSTM con dos miembros: (i) la nueva sLSTM (véase la Sección 2.2) con memoria escalar, actualización escalar y mezcla de memoria, y (ii) la nueva mLSTM
(véase la Sección 2.3) con memoria matricial y una regla de actualización de covarianza (producto externo), totalmente paralelizable. Tanto la sLSTM como la mLSTM mejoran la LSTM mediante la puerta exponencial. Para permitir la paralelización, la mLSTM elimina la mezcla de memoria, es decir, las conexiones recurrentes ocultas-ocultas.
Tanto la mLSTM como la sLSTM pueden extenderse a múltiples celdas de memoria, donde la sLSTM permite la mezcla de memoria entre celdas. Además, el sLSTM puede tener múltiples cabezas sin mezcla de memoria entre ellas, sino solo entre las celdas dentro de cada cabeza. Esta introducción de cabezas para sLSTM, junto con la activación exponencial, establece una nueva forma de mezcla de memoria. Para mLSTM, múltiples cabezas y múltiples celdas son equivalentes.
La integración de estas nuevas variantes de LSTM en módulos de bloques residuales da como resultado bloques xLSTM (véase la Sección 2.4). El apilamiento residual de estos bloques xLSTM en arquitecturas proporciona arquitecturas xLSTM (véase la Sección 2.4). Véase la Figura 1 para la arquitectura xLSTM y sus componentes. 2.1 Revisión de la memoria a largo plazo
La idea original de la LSTM (Hochreiter, 1991; Hochreiter y Schmidhuber, 1997b,a) introdujo la celda de memoria escalar como una unidad central de procesamiento y almacenamiento que evita los gradientes de desaparición (Hochreiter, 1991; Hochreiter et al., 2000) mediante el carrusel de errores constantes (la actualización del estado de la celda). La celda de memoria contiene tres puertas: entrada, salida y olvido. La puerta de olvido fue introducida por Gers et al. (2000). Las reglas de actualización de la celda de memoria de la LSTM en el paso de tiempo t son
Enviar comentarios
Resultados de traducción disponibles
//*//*/**/*//*/


Detectar idioma

inglés

español

Francés

español

inglés

Francés
theoutputofthewholenetworknorthederivativesofthelosswithrespecttotheparameters.
NewMemoryMixing. sLSTMcanhavemultiplememorycells liketheoriginalLSTM(see
AppendixA.2).MultiplememorycellsenablememorymixingviarecurrentconnectionsRz,Ri,
Rf,Rofromhiddenstatevectorhtomemorycellinputzandthegatesi,f,o,respectively.Anew
aspect inmemorymixingistheeffectofexponentialgating.ThenewsLSTMcanhavemultiple
headswithmemorymixingwithineachheadbutnotacrossheads.Theintroductionofheadsfor
sLSTMtogetherwithexponentialgatingestablishesanewwayofmemorymixing.
2.3 mLSTM
ToenhancestoragecapacitiesofLSTMs,weincreasetheLSTMmemorycellfromascalarc∈Rto
amatrixC∈Rd×d.Hence,retrievalisperformedviaamatrixmultiplication.Attimet,wewantto
storeapairofvectors,thekeykt∈Rdandthevaluevt∈Rd(weusetheTransformerterminology).
Laterattimet+τ,thevaluevtshouldberetrievedbyaqueryvectorqt+τ∈Rd.Thisisthesetting
ofBidirectionalAssociativeMemories(BAMs)(Kohonen,1972;Anderson,1972;Nakano,1972;
Andersonetal.,1977).Thecovarianceupdaterule(Sejnowski,1977;Dayan&Willshaw,1991)for
storingakey-valuepairis
Ct =Ct−1+vtk⊤
t . (18)
Weassumealayer-normbeforeprojectinginputstokeysandvalues,thereforetheyhavezeromean.
Thecovarianceupdateruleisoptimal (Dayan&Willshaw,1991)foramaximalseparabilityof
retrievedbinaryvectors,whichisequivalenttoamaximalsignal/noiseratio.Higherseparabilityis
possiblewhenlimitingretrievaltopairwiseinteractionsandconcedingquadraticcomplexitylike
attention(Krotov&Hopfield,2016,2017;Ramsaueretal.,2021).Thecovarianceupdateruleis
equivalenttoFastWeightProgrammers(Schmidhuber,1992;Schlagetal.,2021),whichhavelater
beenequippedwithaconstantdecayratemultipliedtoCt−1andaconstantlearningratemultiplied
tovtk⊤
t (Baetal.,2016a). Inthisspirit,weintegratethecovarianceupdateruleintotheLSTM
framework,wheretheforgetgatecorrespondstodecayrateandtheinputgatetothelearningrate,
whiletheoutputgatescalestheretrievedvector.
Forthismatrixmemory, thenormalizerstateistheweightedsumofkeyvectors,whereeachkey
vectorisweightedbytheinputgateandallfutureforgetgates.Again, thenormalizerstatekeeps
2.051
La salida de toda la red se deriva de las derivadas de la pérdida con respecto a los parámetros.
Nueva mezcla de memoria. El sLSTM puede tener múltiples celdas de memoria como el LSTM original (véase el Apéndice A.2). Múltiples celdas de memoria permiten la mezcla de memoria mediante conexiones recurrentes Rz, Ri, Rf, Ro desde el vector de estado oculto hto la entrada de la celda de memoria z y las puertas i, f, o, respectivamente. Un nuevo aspecto de la mezcla de memoria es el efecto de la puerta exponencial. El nuevo LSTM puede tener múltiples cabezales con mezcla de memoria dentro de cada cabezal, pero no entre ellos. La introducción de cabezales para el sLSTM, junto con la puerta exponencial, establece una nueva forma de mezcla de memoria. 2.3 mLSTM
Para mejorar la capacidad de almacenamiento de los LSTM, aumentamos la celda de memoria del LSTM de un arco escalar ∈ R a una matriz C ∈ Rd × d. Por lo tanto, la recuperación se realiza mediante la multiplicación de matrices. En el instante ∈ m, queremos almacenar un par de vectores, la clave kt ∈ Rd y el valor v ∈ Rd (utilizamos la terminología del transformador). Posteriormente, en el instante t+τ, el valor vt debe recuperarse mediante un vector de consulta qt+τ∈Rd. Esta es la configuración de las Memorias Asociativas Bidireccionales (BAM) (Kohonen, 1972; Anderson, 1972; Nakano, 1972; Anderson et al., 1977). La regla de actualización de covarianza (Sejnowski, 1977; Dayan y Willshaw, 1991) para almacenar un par clave-valor es:
Ct = Ct−1 + vtk⊤
t. (18)
Asumimos una norma de capa antes de proyectar las entradas a claves y valores; por lo tanto, su media es cero. La regla de actualización de covarianza es óptima (Dayan y Willshaw, 1991) para una separabilidad máxima de los vectores binarios recuperados, lo que equivale a una relación señal/ruido máxima. Una mayor separabilidad es posible al limitar la recuperación a interacciones por pares y conceder atención similar a la de la complejidad cuadrática (Krotov y Hopfield, 2016, 2017; Ramsauer et al., 2021). La regla de actualización de covarianza es equivalente a la de Fast Weight Programmers (Schmidhuber, 1992; Schlage et al., 2021), que posteriormente se equiparon con una tasa de decaimiento constante multiplicada por Ct−1 y una tasa de aprendizaje constante multiplicada por vtk⊤t (Bae et al., 2016a). Con este fin, integramos la regla de actualización de covarianza en el marco LSTM, donde la puerta de olvido corresponde a la tasa de decaimiento y la puerta de entrada a la tasa de aprendizaje, mientras que la puerta de salida escala el vector recuperado.
Para esta memoria matricial, el estado del normalizador es la suma ponderada de los vectores clave, donde cada vector clave se pondera por la puerta de entrada y todas las puertas de olvido futuras. De nuevo, el estado del normalizador mantiene
Enviar comentarios
Resultados de traducción disponibles 
