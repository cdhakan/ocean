[Setup]
AppName=CEST-MRF GUI
AppVersion=1.0.0
AppPublisher=CBD Lab
AppPublisherURL=https://github.com/your-lab/cest-mrf
DefaultDirName={autopf}\CEST_MRF_GUI
DefaultGroupName=CEST-MRF GUI
OutputDir=dist
OutputBaseFilename=CEST_MRF_GUI_Setup_v1.0.0
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\CEST_MRF_GUI\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\CEST-MRF GUI";        Filename: "{app}\CEST_MRF_GUI.exe"
Name: "{group}\Uninstall CEST-MRF";  Filename: "{uninstallexe}"
Name: "{commondesktop}\CEST-MRF GUI"; Filename: "{app}\CEST_MRF_GUI.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\CEST_MRF_GUI.exe"; Description: "{cm:LaunchProgram,CEST-MRF GUI}"; Flags: nowait postinstall skipifsilent
