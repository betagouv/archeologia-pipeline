<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="AllStyleCategories">
  <!--
    Style catégorisé par confiance (conf_bin).
    Utilise la palette 0 (Jaune -> Rouge) par défaut.
    
    Les palettes par classe sont appliquées dynamiquement par conversion_shp.py
    selon l'index de la classe (shp_idx % 6):
      0: Jaune -> Orange -> Rouge
      1: Bleu clair -> Bleu foncé
      2: Lavande -> Violet foncé
      3: Vert clair -> Vert foncé
      4: Gris clair -> Gris foncé
      5: Pêche -> Brun
  -->
  <renderer-v2 type="categorizedSymbol" attr="conf_bin" symbollevels="0" enableorderby="0" forceraster="0">
    <categories>
      <category value="[0:0.2[" symbol="0" label="[0:0.2[" render="true"/>
      <category value="[0.2:0.4[" symbol="1" label="[0.2:0.4[" render="true"/>
      <category value="[0.4:0.6[" symbol="2" label="[0.4:0.6[" render="true"/>
      <category value="[0.6:0.8[" symbol="3" label="[0.6:0.8[" render="true"/>
      <category value="[0.8:1]" symbol="4" label="[0.8:1]" render="true"/>
    </categories>
    <symbols>
      <!-- Palette 0: Jaune -> Orange -> Rouge (confiance croissante) -->
      <symbol type="fill" name="0" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" value="255,255,0,255" type="QString"/>
            <Option name="line_style" value="solid" type="QString"/>
            <Option name="line_width" value="0.6" type="QString"/>
            <Option name="line_width_unit" value="MM" type="QString"/>
          </Option>
        </layer>
      </symbol>
      <symbol type="fill" name="1" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" value="255,204,0,255" type="QString"/>
            <Option name="line_style" value="solid" type="QString"/>
            <Option name="line_width" value="0.6" type="QString"/>
            <Option name="line_width_unit" value="MM" type="QString"/>
          </Option>
        </layer>
      </symbol>
      <symbol type="fill" name="2" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" value="255,153,0,255" type="QString"/>
            <Option name="line_style" value="solid" type="QString"/>
            <Option name="line_width" value="0.6" type="QString"/>
            <Option name="line_width_unit" value="MM" type="QString"/>
          </Option>
        </layer>
      </symbol>
      <symbol type="fill" name="3" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" value="255,102,0,255" type="QString"/>
            <Option name="line_style" value="solid" type="QString"/>
            <Option name="line_width" value="0.6" type="QString"/>
            <Option name="line_width_unit" value="MM" type="QString"/>
          </Option>
        </layer>
      </symbol>
      <symbol type="fill" name="4" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" value="255,0,0,255" type="QString"/>
            <Option name="line_style" value="solid" type="QString"/>
            <Option name="line_width" value="0.6" type="QString"/>
            <Option name="line_width_unit" value="MM" type="QString"/>
          </Option>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <selection mode="Default">
    <selectionColor invalid="1"/>
    <selectionSymbol>
      <symbol type="fill" name="" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="color" value="255,255,255,100" type="QString"/>
            <Option name="outline_color" value="255,0,255,255" type="QString"/>
            <Option name="outline_style" value="solid" type="QString"/>
            <Option name="outline_width" value="1.5" type="QString"/>
            <Option name="outline_width_unit" value="MM" type="QString"/>
            <Option name="style" value="solid" type="QString"/>
          </Option>
        </layer>
      </symbol>
    </selectionSymbol>
  </selection>
  <customproperties>
    <Option type="Map">
      <Option name="embeddedWidgets/count" value="0" type="int"/>
    </Option>
  </customproperties>
</qgis>
