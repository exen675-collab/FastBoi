# FastBoi

Prosty interfejs do generowania **wideo z dźwiękiem z opisu tekstowego** za pomocą
FastVideo FastH3 Preview VSA DataFree. Jeden skrypt przygotowuje środowisko Python,
instaluje FastVideo, pobiera kompletny model i uruchamia stronę w przeglądarce.

## Sprzęt — przeczytaj przed wynajmem

**RTX 4060 8 GB nie wystarcza do uruchomienia tego modelu w tym profilu.**
Możesz korzystać z przeglądarki na tym komputerze, a obliczenia wykonywać na Vast.ai.
Instalator wykryje zbyt małą pamięć przed pobraniem wag.

- Autorzy modelu testowali domyślną konfigurację na **4 × B200**.
- Obsługiwany przez instalator system: Linux x86_64 (np. Ubuntu 22.04/24.04)
  lub Ubuntu w WSL2. Wymagany działający sterownik NVIDIA; skrypt go nie instaluje.
- Inne GPU: profil `portable`, NVIDIA Ampere lub nowsza, CUDA 12.6/13.0.
  Liczba kart musi dzielić 56. Domyślnie używana jest jedna karta;
  wiele kart włączasz przez `--num-gpus`.
- Próg 60 GB VRAM łącznie jest tylko wstępnym filtrem instalatora, **nie minimalną
  konfiguracją potwierdzoną testami**. Pamięć aktywacji, enkodera i VAE może wymagać
  znacznie więcej. Nie zakładaj, że dowolny zestaw kart sumujący się do 60 GB zadziała.
- Kompletny snapshot ma około **148 GB** (wagi i komponenty). Przeznacz orientacyjnie
  250 GB dysku na środowisko, cache i wyniki. Instalator sprawdza miejsce na brakujące pliki.
  Offload używa również RAM hosta; dobierz duży zapas RAM (orientacyjnie 256 GB lub więcej,
  nie jest to zweryfikowane minimum).
- Internet jest wymagany do pierwszego pobrania. Pliki są przechowywane w cache.

Źródła: [karta modelu](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree),
[instalacja FastVideo](https://github.com/hao-ai-lab/FastVideo/blob/2413a57651a9720e820b64dddac9de4450a1164b/docs/getting_started/installation/gpu.md),
[oficjalny przykład FastH3](https://github.com/hao-ai-lab/FastVideo/blob/2413a57651a9720e820b64dddac9de4450a1164b/examples/inference/basic/basic_fasth3.py).

## Start na Vast.ai

Wynajmij instancję Linux z odpowiednim GPU, pamięcią RAM i dyskiem. Wybierz dostęp SSH
i dodaj klucz SSH do konta. Połącz się komendą z panelu Vast.ai, dodając przekierowanie:

```bash
ssh -p PORT_SSH root@HOST -L 7860:127.0.0.1:7860
```

Zastąp `PORT_SSH` i `HOST` danymi swojej instancji. W jej terminalu:

```bash
git clone TWOJ_ADRES_REPO FastBoi
cd FastBoi
bash start.sh --num-gpus 4 --profile blackwell --no-browser
```

`blackwell` jest przeznaczony dla B200/GB200 (SM100) i sterownika z obsługą CUDA 13.
Dla innych GPU użyj np. `bash start.sh --num-gpus 2 --no-browser` — tylko jeżeli
wybrany sprzęt ma wystarczającą pamięć. Domyślny `portable` używa VSA Triton,
wyłącza FA4 i kompilację oraz włącza sharding DiT przy wielu kartach.

Po komunikacie o gotowym serwerze otwórz na swoim komputerze
[http://localhost:7860](http://localhost:7860). Pozostaw tunel SSH aktywny.
Serwer warto uruchomić w sesji `tmux`, aby rozłączenie terminala nie przerwało pracy.

Alternatywnie wystaw port **7860/TCP** w konfiguracji instancji Vast.ai i uruchom:

```bash
bash start.sh --num-gpus 4 --profile blackwell --host 0.0.0.0 --no-browser
```

Aplikacja wypisze login i losowe hasło dla tej sesji. Możesz ustawić własne przez
`FASTBOI_USER` i `FASTBOI_PASSWORD`. Użyj HTTPS z Instance Portal lub tunelu SSH;
bezpośredni HTTP na publicznym porcie nie szyfruje hasła ani treści.
Adres i publiczny port odczytaj z mapowania w panelu Vast.ai — publiczny port
nie musi wynosić 7860. Skrypt nie wynajmuje instancji ani nie zmienia ustawień konta.

[Instrukcja tunelu SSH Vast.ai](https://docs.vast.ai/guides/instances/connect/ssh) ·
[Mapowanie portów](https://docs.vast.ai/guides/instances/connect/networking)

## Start lokalny

Na kompatybilnej maszynie z Linuxem:

```bash
bash start.sh
```

W Windows najpierw zainstaluj Ubuntu WSL2 (`wsl --install -d Ubuntu`), skonfiguruj
konto Linux i upewnij się, że Ubuntu jest domyślną dystrybucją (`wsl --set-default Ubuntu`).
Docker Desktop nie zastępuje dystrybucji Ubuntu. Następnie w PowerShell:

```powershell
.\start.ps1
```

Ze względu na wydajność dużych plików najlepiej klonować repozytorium bezpośrednio
w systemie plików Ubuntu (np. `~/FastBoi`) i uruchamiać `bash start.sh` w WSL.
Przeglądarka otwiera się automatycznie tam, gdzie środowisko na to pozwala;
w WSL można ręcznie otworzyć localhost:7860 w Windows.
Na RTX 4060 kontrola sprzętu zatrzyma uruchomienie z wyjaśnieniem.

## Co robi skrypt

1. Uzupełnia brakujące narzędzia systemowe przez apt (na Ubuntu/Debian; może wymagać sudo).
2. Sprawdza GPU, instaluje `uv` w razie potrzeby i tworzy `.venv` z Pythonem 3.12.
3. Pobiera przypięty commit FastVideo z `config.json` do `.runtime/FastVideo`.
4. Instaluje wariant PyTorch dobrany do sterownika, sprawdza zależności i widoczność GPU.
5. Pobiera przypięty snapshot Hugging Face, w tym enkoder tekstu, VAE i tokenizer.
6. Uruchamia interfejs. Model ładuje się do pamięci przy pierwszej generacji i pozostaje
   w niej do zamknięcia aplikacji. Kolejne żądania wykonują się pojedynczo (kolejka do 8).

Ponowne uruchomienie pomija zakończoną instalację zależności i korzysta z cache modelu.
Przerwane pobieranie można ponowić tym samym poleceniem.
`HF_HOME` pozwala przenieść cache na trwały wolumen. `HF_TOKEN` służy do autoryzacji
Hugging Face, jeśli dostęp będzie wymagał konta lub zaakceptowania licencji.
Nigdy nie zapisuj tokenu w repozytorium.

W interfejsie: opis sceny, trzy formaty, seed, podgląd i pobranie MP4 oraz JSON
z parametrami. Domyślna długość to 124 klatki przy 24 FPS. Oficjalny parametr
`steps=5` oznacza **pięć punktów harmonogramu, czyli cztery przebiegi transformera**.
To model preview text-to-audio-video; interfejs nie obsługuje wejściowych obrazów.

Wyniki zapisują się w `outputs/<id>/`; nie są automatycznie kasowane.
Przed usunięciem wynajętej instancji pobierz wyniki i zabezpiecz dane na wolumenie.

## Diagnostyka i zakres weryfikacji

```bash
bash start.sh --check
python -m unittest discover -s tests -v
```

Przy błędzie CUDA/OOM szczegóły znajdują się w terminalu. Zmniejszenie formatu może
ograniczyć pamięć aktywacji, ale nie zmniejsza samych wag modelu. Po błędzie silnik
jest zamykany; kolejna próba ładuje go ponownie. Przy awarii procesu CUDA uruchom serwer ponownie.
Sterownik musi obsługiwać wybrany build PyTorch; instalator nie aktualizuje sterownika hosta.

Testy bez GPU sprawdzają kontrolę sprzętu, walidację żądań, unikalność wyników, metadane i reset silnika
po błędzie. Interfejs uruchomiono i sprawdzono w przeglądarce, w tym obsługę pustego opisu.
**Pełna instalacja FastVideo i rzeczywista generacja wymagają testu na docelowym
GPU; nie zostały wykonane na lokalnej RTX 4060.** Przypięte są rewizje FastVideo i modelu;
zależności przechodnie nadal rozwiązuje uv według wymagań FastVideo.

Model podlega MiniMax H3 Community License wskazanej na stronie Hugging Face.
